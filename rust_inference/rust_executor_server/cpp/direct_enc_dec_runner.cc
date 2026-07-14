#include "direct_enc_dec_runner.h"

#include "direct_cache_tile.h"
#include "direct_dynamic_decode.h"
#include "tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/kernels/decodingKernels.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <string>

namespace tr = tensorrt_llm::runtime;
namespace tk = tensorrt_llm::kernels;
namespace fs = std::filesystem;

namespace
{
constexpr int32_t kDecoderStartId = 2;
constexpr int32_t kEosId = 2;
constexpr int32_t kPadId = 1;
constexpr int32_t kHiddenSize = 256;
constexpr int32_t kNumLayers = 6;
constexpr int32_t kNumHeads = 4;
constexpr int32_t kHeadSize = 64;
constexpr int32_t kVocabSize = 806;
constexpr int32_t kVocabSizePadded = 806;

void checkCuda(cudaError_t status, char const* what)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

std::string readFile(fs::path const& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("failed to open " + path.string());
    }
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

int32_t jsonInt(std::string const& text, std::string const& key, int32_t fallback)
{
    std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (std::regex_search(text, match, re))
    {
        return std::stoi(match[1].str());
    }
    return fallback;
}

struct Logger : nvinfer1::ILogger
{
    void log(Severity severity, char const* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            fprintf(stderr, "[TensorRT] %s\n", msg);
        }
    }
};

struct DeviceBuffer
{
    void* ptr{nullptr};
    size_t bytes{0};

    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) { resize(n); }
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr(other.ptr)
        , bytes(other.bytes)
    {
        other.ptr = nullptr;
        other.bytes = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            release();
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
    ~DeviceBuffer() { release(); }

    void resize(size_t n)
    {
        if (n <= bytes)
        {
            return;
        }
        release();
        bytes = n;
        checkCuda(cudaMalloc(&ptr, bytes), "cudaMalloc");
    }

    void release()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
        ptr = nullptr;
        bytes = 0;
    }

    template <typename T>
    T* as()
    {
        return static_cast<T*>(ptr);
    }
};

nvinfer1::Dims dims(std::initializer_list<int64_t> values)
{
    nvinfer1::Dims out{};
    out.nbDims = static_cast<int32_t>(values.size());
    int32_t i = 0;
    for (auto value : values)
    {
        out.d[i++] = value;
    }
    return out;
}

void setShape(nvinfer1::IExecutionContext& ctx, char const* name, nvinfer1::Dims shape)
{
    if (!ctx.setInputShape(name, shape))
    {
        throw std::runtime_error(std::string("failed to set input shape for ") + name);
    }
}

void setAddr(nvinfer1::IExecutionContext& ctx, char const* name, void* ptr)
{
    if (!ctx.setTensorAddress(name, ptr))
    {
        throw std::runtime_error(std::string("failed to set tensor address for ") + name);
    }
}

tr::ITensor::SharedPtr wrapTensor(void* ptr, nvinfer1::DataType type, nvinfer1::Dims shape)
{
    return tr::ITensor::SharedPtr(tr::ITensor::wrap(ptr, type, shape).release());
}

} // namespace

namespace indicxlit
{

struct DirectEncDecRunner::Impl
{
    Impl(fs::path const& engineDir, int32_t requestedMaxBatchSize, int32_t requestedMaxBeamWidth, int32_t requestedMaxNewTokens)
        : runtime(nvinfer1::createInferRuntime(logger))
    {
        if (!runtime)
        {
            throw std::runtime_error("failed to create TensorRT runtime");
        }

        auto decoderConfig = readFile(engineDir / "decoder" / "config.json");
        maxBatchSize = std::min(requestedMaxBatchSize, jsonInt(decoderConfig, "max_batch_size", requestedMaxBatchSize));
        maxBeamWidth = std::min(requestedMaxBeamWidth, jsonInt(decoderConfig, "max_beam_width", requestedMaxBeamWidth));
        maxSeqLen = jsonInt(decoderConfig, "max_seq_len", requestedMaxNewTokens + 1);
        maxNewTokens = std::min(requestedMaxNewTokens, maxSeqLen - 1);
        auto encoderConfig = readFile(engineDir / "encoder" / "config.json");
        maxEncoderLen = std::min(jsonInt(decoderConfig, "max_encoder_input_len", 128), jsonInt(encoderConfig, "max_input_len", 128));

        encoderEngine = loadEngine(engineDir / "encoder" / "rank0.engine");
        decoderEngine = loadEngine(engineDir / "decoder" / "rank0.engine");
        encoderContext.reset(encoderEngine->createExecutionContext());
        decoderContext.reset(decoderEngine->createExecutionContext());
        if (!encoderContext || !decoderContext)
        {
            throw std::runtime_error("failed to create TensorRT execution contexts");
        }

        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        decoder = std::make_unique<DynamicDecodeAdapter>(maxBatchSize, maxBeamWidth, kVocabSize, kVocabSizePadded, stream);
        cudaGraphDecode = std::getenv("INDICXLIT_CUDA_GRAPH_DECODE") != nullptr;
        if (auto const* value = std::getenv("INDICXLIT_GRAPH_BATCH_SIZE"))
        {
            graphBatchSize = std::atoi(value);
        }
        if (auto const* value = std::getenv("INDICXLIT_GRAPH_ENCODER_LEN_BUCKET"))
        {
            graphEncoderLenBucket = std::atoi(value);
        }
        if (auto const* value = std::getenv("INDICXLIT_GRAPH_CHUNK_SIZE"))
        {
            graphChunkSize = std::atoi(value);
        }
        graphBatchSize = std::clamp(graphBatchSize, 1, maxBatchSize);
        graphEncoderLenBucket = std::clamp(graphEncoderLenBucket, 1, maxEncoderLen);
        graphChunkSize = std::max(1, graphChunkSize);
    }

    ~Impl()
    {
        clearDecodeGraphs();
        if (stream)
        {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    std::unique_ptr<nvinfer1::ICudaEngine> loadEngine(fs::path const& path)
    {
        auto buffer = readFile(path);
        auto* engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
        if (!engine)
        {
            throw std::runtime_error("failed to deserialize TensorRT engine " + path.string());
        }
        return std::unique_ptr<nvinfer1::ICudaEngine>(engine);
    }

    struct DecodeGraph
    {
        cudaGraph_t graph{};
        cudaGraphExec_t exec{};
    };

    void clearDecodeGraphs()
    {
        for (auto& item : decodeGraphs)
        {
            if (item.second.exec)
            {
                cudaGraphExecDestroy(item.second.exec);
            }
            if (item.second.graph)
            {
                cudaGraphDestroy(item.second.graph);
            }
        }
        decodeGraphs.clear();
        decodeGraphWarmupDone = false;
    }

    void allocateFor(int32_t batchSize, int32_t beamWidth, int32_t currentMaxEncoderLen, int32_t totalEncoderTokens)
    {
        auto const maxRows = static_cast<size_t>(batchSize) * static_cast<size_t>(beamWidth);
        auto const maxEncTokens = static_cast<size_t>(std::max(1, totalEncoderTokens));
        auto const crossMaskTokens = static_cast<size_t>(std::max(1, currentMaxEncoderLen));
        auto const crossCacheLen = static_cast<size_t>(std::max(1, currentMaxEncoderLen));
        auto const crossPackedCols = static_cast<size_t>(((std::max(1, currentMaxEncoderLen) + 255) / 256) * 8);
        auto const crossPackedContextRows = static_cast<size_t>(batchSize) * 128ULL;
        auto const kvElems = maxRows * 2ULL * kNumHeads * static_cast<size_t>(maxSeqLen) * kHeadSize;
        auto const crossKvElems = maxRows * 2ULL * kNumHeads * crossCacheLen * kHeadSize;

        encoderInputIds.resize(maxEncTokens * sizeof(int32_t));
        encoderPositionIds.resize(maxEncTokens * sizeof(int32_t));
        encoderInputLengths.resize(maxRows * sizeof(int32_t));
        encoderMaxInputLength.resize(crossCacheLen * sizeof(int32_t));
        encoderOutput.resize(maxEncTokens * kHiddenSize * sizeof(half));

        inputIds.resize(maxRows * sizeof(int32_t));
        positionIds.resize(maxRows * sizeof(int32_t));
        encoderInputLengthsTiled.resize(maxRows * sizeof(int32_t));
        decoderScalarInts.resize(maxRows * sizeof(int32_t));
        contextLengths.resize(maxRows * sizeof(int32_t));
        sequenceLength.resize(maxRows * sizeof(int32_t));
        sequenceLimitLength.resize(static_cast<size_t>(batchSize) * sizeof(int32_t));
        requestTypes.resize(maxRows * sizeof(int32_t));
        lastTokenIds.resize(maxRows * sizeof(int32_t));
        crossKvCacheGen.resize(sizeof(uint8_t));
        crossAttentionMask.resize(maxRows * crossMaskTokens * sizeof(uint8_t));
        crossAttentionCuQSeqLens.resize((static_cast<size_t>(batchSize) + 1ULL) * sizeof(int32_t));
        crossAttentionCuMaskRows.resize((static_cast<size_t>(batchSize) + 1ULL) * sizeof(int32_t));
        auto const crossPackedGenerationElems = static_cast<size_t>(batchSize) * ((crossMaskTokens + 31) / 32);
        crossAttentionPackedMask.resize(
            std::max<size_t>(crossPackedContextRows * crossPackedCols, crossPackedGenerationElems) * sizeof(int32_t));
        cacheIndirection[0].resize(maxRows * static_cast<size_t>(maxSeqLen) * sizeof(int32_t));
        cacheIndirection[1].resize(maxRows * static_cast<size_t>(maxSeqLen) * sizeof(int32_t));
        logits.resize(maxRows * kVocabSizePadded * sizeof(half));
        logitsTiled.resize(maxRows * kVocabSizePadded * sizeof(half));
        outputIds.resize(maxRows * static_cast<size_t>(maxSeqLen) * sizeof(int32_t));
        newTokens.resize(maxRows * sizeof(int32_t));
        finished.resize(maxRows * sizeof(uint8_t));
        parentIds.resize(maxRows * static_cast<size_t>(maxSeqLen) * sizeof(int32_t));
        cumLogProbs.resize(maxRows * sizeof(float));
        finalizedOutputIds.resize(maxRows * static_cast<size_t>(maxSeqLen) * sizeof(int32_t));
        lengthPenalty.resize(static_cast<size_t>(batchSize) * sizeof(float));
        beamHypsOutputIdsCba.resize(static_cast<size_t>(batchSize) * beamWidth * 2ULL * maxSeqLen * sizeof(int32_t));
        beamHypsSeqLenCba.resize(static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(int32_t));
        beamHypsCumLogProbsCba.resize(static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(float));
        beamHypsNormedScoresCba.resize(static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(float));
        beamHypsLogProbsCba.resize(static_cast<size_t>(batchSize) * beamWidth * 2ULL * maxSeqLen * sizeof(float));
        beamHypsMinNormedScores.resize(static_cast<size_t>(batchSize) * sizeof(float));
        beamHypsNumBeams.resize(static_cast<size_t>(batchSize) * sizeof(int32_t));
        beamHypsIsDone.resize(static_cast<size_t>(batchSize) * sizeof(uint8_t));

        for (int i = 0; i < kNumLayers; ++i)
        {
            kvContext[i].resize(static_cast<size_t>(maxBatchSize) * 2ULL * kNumHeads * static_cast<size_t>(maxSeqLen) * kHeadSize * sizeof(half));
            kv[i].resize(kvElems * sizeof(half));
            crossKvContext[i].resize(static_cast<size_t>(batchSize) * 2ULL * kNumHeads * crossCacheLen * kHeadSize * sizeof(half));
            crossKv[i].resize(crossKvElems * sizeof(half));
            hostTileLayerPtrs[i] = TileLayerPtrs{
                kvContext[i].as<half>(),
                kv[i].as<half>(),
                crossKvContext[i].as<half>(),
                crossKv[i].as<half>(),
            };
        }
        tileLayerPtrs.resize(hostTileLayerPtrs.size() * sizeof(TileLayerPtrs));
        copyHostToDevice(tileLayerPtrs, hostTileLayerPtrs.data(), hostTileLayerPtrs.size() * sizeof(TileLayerPtrs));
    }

    void copyHostToDevice(DeviceBuffer& dst, void const* src, size_t bytes)
    {
        checkCuda(cudaMemcpyAsync(dst.ptr, src, bytes, cudaMemcpyHostToDevice, stream), "cudaMemcpyAsync H2D");
    }

    void memsetDevice(DeviceBuffer& dst, int value, size_t bytes)
    {
        checkCuda(cudaMemsetAsync(dst.ptr, value, bytes, stream), "cudaMemsetAsync");
    }

    void runEncoder(
        std::vector<std::vector<int32_t>> const& inputs, int32_t& totalTokens, int32_t& maxInputLen, int32_t forcedMaxInputLen = 0)
    {
        totalTokens = 0;
        maxInputLen = 0;
        hostFlatIds.clear();
        hostFlatPositions.clear();
        hostEncoderLengths.clear();
        hostEncoderLengths.reserve(inputs.size());
        for (auto const& input : inputs)
        {
            auto const len = static_cast<int32_t>(input.size());
            hostEncoderLengths.push_back(len);
            maxInputLen = std::max(maxInputLen, len);
            hostFlatIds.insert(hostFlatIds.end(), input.begin(), input.end());
            for (int32_t i = 0; i < len; ++i)
            {
                hostFlatPositions.push_back(i);
            }
            totalTokens += len;
        }

        auto const effectiveMaxInputLen = std::max(std::max(1, maxInputLen), forcedMaxInputLen);
        hostMaxLenShape.assign(effectiveMaxInputLen, 0);
        copyHostToDevice(encoderInputIds, hostFlatIds.data(), hostFlatIds.size() * sizeof(int32_t));
        copyHostToDevice(encoderPositionIds, hostFlatPositions.data(), hostFlatPositions.size() * sizeof(int32_t));
        copyHostToDevice(encoderInputLengths, hostEncoderLengths.data(), hostEncoderLengths.size() * sizeof(int32_t));
        copyHostToDevice(encoderMaxInputLength, hostMaxLenShape.data(), hostMaxLenShape.size() * sizeof(int32_t));

        auto& ctx = *encoderContext;
        setShape(ctx, "input_ids", dims({totalTokens}));
        setShape(ctx, "position_ids", dims({totalTokens}));
        setShape(ctx, "input_lengths", dims({static_cast<int64_t>(inputs.size())}));
        setShape(ctx, "max_input_length", dims({effectiveMaxInputLen}));
        setAddr(ctx, "input_ids", encoderInputIds.ptr);
        setAddr(ctx, "position_ids", encoderPositionIds.ptr);
        setAddr(ctx, "input_lengths", encoderInputLengths.ptr);
        setAddr(ctx, "max_input_length", encoderMaxInputLength.ptr);
        setAddr(ctx, "encoder_output", encoderOutput.ptr);
        if (!ctx.enqueueV3(stream))
        {
            throw std::runtime_error("encoder enqueueV3 failed");
        }
    }

    void prepareCommon(int32_t batchSize, int32_t beamWidth, int32_t activeRows, int32_t totalEncoderTokens,
        int32_t maxEncoderInputLen, int32_t step, bool contextPhase)
    {
        auto const decodeRows = contextPhase && beamWidth > 1 ? batchSize * beamWidth : activeRows;
        hostDecodeOnes.assign(decodeRows, 1);
        hostPositions.assign(activeRows, contextPhase ? 0 : step + 1);
        hostRequestTypes.assign(activeRows, contextPhase ? 0 : 1);
        hostLastTokenIds.resize(activeRows);
        std::iota(hostLastTokenIds.begin(), hostLastTokenIds.end(), 1);
        hostPastKeyValueLengths.assign(activeRows, contextPhase ? 1 : 1 + step);
        hostMaxAttentionWindowSizes.assign(kNumLayers, maxSeqLen);
        hostRuntimePerfKnobs.assign(16, -1);
        hostRuntimePerfKnobs[0] = 1;
        hostContextProgress.assign(1, 0);
        hostSinkTokenLength.assign(1, 0);
        hostContextLengths.assign(activeRows, 1);

        if (contextPhase)
        {
            copyHostToDevice(inputIds, contextStartIds.data(), activeRows * sizeof(int32_t));
            decoderInputIdsPtr = inputIds.ptr;
        }
        else
        {
            decoderInputIdsPtr = newTokens.ptr;
        }
        if (contextPhase)
        {
            memsetDevice(cacheIndirection[0], 0, static_cast<size_t>(batchSize) * beamWidth * maxSeqLen * sizeof(int32_t));
            memsetDevice(cacheIndirection[1], 0, static_cast<size_t>(batchSize) * beamWidth * maxSeqLen * sizeof(int32_t));
            memsetDevice(outputIds, 0, static_cast<size_t>(batchSize) * beamWidth * maxSeqLen * sizeof(int32_t));
            memsetDevice(parentIds, 0, static_cast<size_t>(batchSize) * beamWidth * maxSeqLen * sizeof(int32_t));
            memsetDevice(finished, 0, static_cast<size_t>(batchSize) * beamWidth * sizeof(uint8_t));
            memsetDevice(beamHypsOutputIdsCba, 0, static_cast<size_t>(batchSize) * beamWidth * 2ULL * maxSeqLen * sizeof(int32_t));
            memsetDevice(beamHypsSeqLenCba, 0, static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(int32_t));
            memsetDevice(beamHypsCumLogProbsCba, 0, static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(float));
            memsetDevice(beamHypsNormedScoresCba, 0, static_cast<size_t>(batchSize) * beamWidth * 2ULL * sizeof(float));
            memsetDevice(beamHypsLogProbsCba, 0, static_cast<size_t>(batchSize) * beamWidth * 2ULL * maxSeqLen * sizeof(float));
            memsetDevice(beamHypsMinNormedScores, 0, static_cast<size_t>(batchSize) * sizeof(float));
            memsetDevice(beamHypsNumBeams, 0, static_cast<size_t>(batchSize) * sizeof(int32_t));
            memsetDevice(beamHypsIsDone, 0, static_cast<size_t>(batchSize) * sizeof(uint8_t));
            hostCumLogProbs.assign(static_cast<size_t>(batchSize) * beamWidth, -1.0e20f);
            for (int32_t b = 0; b < batchSize; ++b)
            {
                hostCumLogProbs[static_cast<size_t>(b) * beamWidth] = 0.0f;
            }
            copyHostToDevice(cumLogProbs, hostCumLogProbs.data(), hostCumLogProbs.size() * sizeof(float));
        }

        auto const maskCols = std::max(1, maxEncoderInputLen);
        auto const packedColsContext = ((maskCols + 255) / 256) * 8;
        auto const packedColsGeneration = std::max(1, (maskCols + 31) / 32);
        auto const packedRows = contextPhase ? batchSize * 128 : batchSize;
        auto const packedCols = contextPhase ? packedColsContext : packedColsGeneration;
        auto const* maskLengths = contextPhase ? encoderInputLengths.as<int32_t>() : encoderInputLengthsTiled.as<int32_t>();
        checkCuda(launchPrepareDecodeInputs(positionIds.as<int32_t>(), requestTypes.as<int32_t>(),
                      lastTokenIds.as<int32_t>(), contextLengths.as<int32_t>(), sequenceLength.as<int32_t>(),
                      sequenceLimitLength.as<int32_t>(), crossKvCacheGen.as<uint8_t>(), crossAttentionMask.as<uint8_t>(),
                      maskLengths, activeRows, decodeRows, batchSize, maskCols, contextPhase ? 0 : step + 1,
                      contextPhase ? 0 : 1, maxSeqLen, contextPhase, stream),
            "launchPrepareDecodeInputs");
        if (contextPhase)
        {
            tk::PackedMaskParams<bool> maskParams{};
            maskParams.maskInput = static_cast<bool const*>(crossAttentionMask.ptr);
            maskParams.cuQSeqLens = static_cast<int32_t*>(crossAttentionCuQSeqLens.ptr);
            maskParams.packedMask = static_cast<uint32_t*>(crossAttentionPackedMask.ptr);
            maskParams.cuMaskRows = static_cast<int32_t*>(crossAttentionCuMaskRows.ptr);
            maskParams.actualQSeqLens = static_cast<int32_t const*>(contextLengths.ptr);
            maskParams.actualKvSeqLens = static_cast<int32_t const*>(encoderInputLengths.ptr);
            maskParams.batchSize = batchSize;
            maskParams.maxQSeqLen = 1;
            maskParams.maxKvSeqLen = maskCols;
            maskParams.attentionMaskType = tk::ContextAttentionMaskType::CUSTOM_MASK;
            maskParams.validPosVal = true;
            tk::invokeBuildPackedMask(maskParams, stream);
        }
        else
        {
            memsetDevice(crossAttentionPackedMask, 0, static_cast<size_t>(packedRows) * packedCols * sizeof(int32_t));
        }

        (void) maxEncoderInputLen;
        (void) totalEncoderTokens;
    }

    void setDecoderBindings(int32_t activeRows, int32_t batchSize, int32_t beamWidth, int32_t totalEncoderTokens,
        int32_t maxEncoderInputLen, int32_t step, bool contextPhase)
    {
        auto& ctx = *decoderContext;
        auto const encoderRows = contextPhase ? totalEncoderTokens : 0;
        auto const maskCols = std::max(1, maxEncoderInputLen);
        auto const packedRows = contextPhase ? batchSize * 128 : batchSize;
        auto const packedCols = contextPhase ? ((maskCols + 255) / 256) * 8 : std::max(1, (maskCols + 31) / 32);
        auto const kvRows = contextPhase ? batchSize : batchSize * beamWidth;
        auto* kvBase = contextPhase ? kvContext : kv;
        auto* crossKvBase = contextPhase ? crossKvContext : crossKv;
        auto* encoderLengths = contextPhase ? encoderInputLengths.ptr : encoderInputLengthsTiled.ptr;
        int32_t const cacheSrcIndex = step % 2 == 0 ? 0 : 1;

        if (!contextPhase && generationBindingsValid && generationBindingBatchSize == batchSize
            && generationBindingBeamWidth == beamWidth && generationBindingMaxEncoderInputLen == maxEncoderInputLen)
        {
            setAddr(ctx, "cache_indirection", cacheIndirection[cacheSrcIndex].ptr);
            return;
        }

        setShape(ctx, "input_ids", dims({activeRows}));
        setShape(ctx, "position_ids", dims({activeRows}));
        setShape(ctx, "encoder_input_lengths", dims({activeRows}));
        setShape(ctx, "encoder_max_input_length", dims({maxEncoderInputLen}));
        setShape(ctx, "encoder_output", dims({encoderRows, kHiddenSize}));
        setShape(ctx, "host_past_key_value_lengths", dims({activeRows}));
        setShape(ctx, "host_context_lengths", dims({activeRows}));
        setShape(ctx, "sequence_length", dims({activeRows}));
        setShape(ctx, "context_lengths", dims({activeRows}));
        setShape(ctx, "host_request_types", dims({activeRows}));
        setShape(ctx, "last_token_ids", dims({activeRows}));
        setShape(ctx, "cross_attention_mask", dims({activeRows, maskCols}));
        setShape(ctx, "cross_attention_packed_mask", dims({packedRows, packedCols}));
        setShape(ctx, "cache_indirection", dims({batchSize, beamWidth, maxSeqLen}));

        setAddr(ctx, "input_ids", decoderInputIdsPtr);
        setAddr(ctx, "position_ids", positionIds.ptr);
        setAddr(ctx, "encoder_input_lengths", encoderLengths);
        setAddr(ctx, "encoder_max_input_length", encoderMaxInputLength.ptr);
        setAddr(ctx, "encoder_output", encoderOutput.ptr);
        setAddr(ctx, "host_past_key_value_lengths", hostPastKeyValueLengths.data());
        setAddr(ctx, "host_context_lengths", hostContextLengths.data());
        setAddr(ctx, "sequence_length", sequenceLength.ptr);
        setAddr(ctx, "context_lengths", contextLengths.ptr);
        setAddr(ctx, "host_request_types", hostRequestTypes.data());
        setAddr(ctx, "host_runtime_perf_knobs", hostRuntimePerfKnobs.data());
        setAddr(ctx, "host_context_progress", hostContextProgress.data());
        setAddr(ctx, "last_token_ids", lastTokenIds.ptr);
        setAddr(ctx, "cross_attention_mask", crossAttentionMask.ptr);
        setAddr(ctx, "cross_attention_packed_mask", crossAttentionPackedMask.ptr);
        setAddr(ctx, "cache_indirection", cacheIndirection[cacheSrcIndex].ptr);
        setAddr(ctx, "host_max_attention_window_sizes", hostMaxAttentionWindowSizes.data());
        setAddr(ctx, "host_sink_token_length", hostSinkTokenLength.data());
        setAddr(ctx, "cross_kv_cache_gen", crossKvCacheGen.ptr);
        setAddr(ctx, "logits", logits.ptr);

        for (int32_t layer = 0; layer < kNumLayers; ++layer)
        {
            auto kvShape = dims({kvRows, 2, kNumHeads, maxSeqLen, kHeadSize});
            auto crossShape = dims({kvRows, 2, kNumHeads, maxEncoderInputLen, kHeadSize});
            auto pkv = "past_key_value_" + std::to_string(layer);
            auto present = "present_key_value_" + std::to_string(layer);
            auto crossPast = "cross_past_key_value_" + std::to_string(layer);
            auto crossPresent = "cross_present_key_value_" + std::to_string(layer);
            setShape(ctx, pkv.c_str(), kvShape);
            setShape(ctx, crossPast.c_str(), crossShape);
            setAddr(ctx, pkv.c_str(), kvBase[layer].ptr);
            setAddr(ctx, present.c_str(), kvBase[layer].ptr);
            setAddr(ctx, crossPast.c_str(), crossKvBase[layer].ptr);
            setAddr(ctx, crossPresent.c_str(), crossKvBase[layer].ptr);
        }

        (void) step;

        if (!contextPhase)
        {
            generationBindingsValid = true;
            generationBindingBatchSize = batchSize;
            generationBindingBeamWidth = beamWidth;
            generationBindingMaxEncoderInputLen = maxEncoderInputLen;
        }
        else
        {
            generationBindingsValid = false;
        }
    }

    void tileEncoderLengths(std::vector<int32_t> const& lengths, int32_t beamWidth)
    {
        hostTiledEncoderLengths.clear();
        hostTiledEncoderLengths.reserve(lengths.size() * static_cast<size_t>(beamWidth));
        for (auto len : lengths)
        {
            for (int32_t beam = 0; beam < beamWidth; ++beam)
            {
                hostTiledEncoderLengths.push_back(len);
            }
        }
        copyHostToDevice(
            encoderInputLengthsTiled, hostTiledEncoderLengths.data(), hostTiledEncoderLengths.size() * sizeof(int32_t));
    }

    void setupDecoder(int32_t batchSize, int32_t beamWidth)
    {
        bool const cumLogProbs = beamWidth > 1;
        if (decoderSetupValid && decoderSetupBatchSize == batchSize && decoderSetupBeamWidth == beamWidth
            && decoderSetupCumLogProbs == cumLogProbs)
        {
            return;
        }
        decoder->setup(batchSize, beamWidth, false, cumLogProbs);
        decoderSetupValid = true;
        decoderSetupBatchSize = batchSize;
        decoderSetupBeamWidth = beamWidth;
        decoderSetupCumLogProbs = cumLogProbs;
    }

    void tileContextCaches(int32_t batchSize, int32_t beamWidth, int32_t maxEncoderInputLen)
    {
        if (beamWidth == 1)
        {
            for (int32_t layer = 0; layer < kNumLayers; ++layer)
            {
                checkCuda(cudaMemcpyAsync(kv[layer].ptr, kvContext[layer].ptr,
                              static_cast<size_t>(batchSize) * 2ULL * kNumHeads * maxSeqLen * kHeadSize * sizeof(half),
                              cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpyAsync tile self kv");
                checkCuda(cudaMemcpyAsync(crossKv[layer].ptr, crossKvContext[layer].ptr,
                              static_cast<size_t>(batchSize) * 2ULL * kNumHeads * maxEncoderInputLen * kHeadSize * sizeof(half),
                              cudaMemcpyDeviceToDevice, stream),
                    "cudaMemcpyAsync tile cross kv");
            }
            return;
        }

        auto const selfBytes = 2ULL * kNumHeads * static_cast<size_t>(maxSeqLen) * kHeadSize * sizeof(half);
        auto const crossBytes = 2ULL * kNumHeads * static_cast<size_t>(maxEncoderInputLen) * kHeadSize * sizeof(half);
        auto const selfElems = selfBytes / sizeof(half);
        auto const crossElems = crossBytes / sizeof(half);
        checkCuda(launchTileBeamCopyAllLayers(tileLayerPtrs.as<TileLayerPtrs>(), kNumLayers, selfElems, crossElems,
                      batchSize, beamWidth, stream),
            "launchTileBeamCopyAllLayers kv");
    }

    void tileLogits(int32_t batchSize, int32_t beamWidth)
    {
        if (beamWidth == 1)
        {
            return;
        }
        auto const rowBytes = static_cast<size_t>(kVocabSizePadded) * sizeof(half);
        checkCuda(launchTileBeamCopyHalf(
                      logits.as<half>(), logitsTiled.as<half>(), batchSize, beamWidth, rowBytes / sizeof(half), stream),
            "launchTileBeamCopyHalf logits");
    }

    bool runDecodeStep(int32_t batchSize, int32_t beamWidth, int32_t totalEncoderTokens, int32_t maxEncoderInputLen,
        int32_t step, bool contextPhase, bool refreshEndIds = true, bool forceFinishedCheck = false,
        bool deferFinishedSync = false, bool useDecodeGraph = false)
    {
        auto const activeRows = contextPhase ? batchSize : batchSize * beamWidth;
        prepareCommon(batchSize, beamWidth, activeRows, totalEncoderTokens, maxEncoderInputLen, step, contextPhase);
        setDecoderBindings(activeRows, batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, step, contextPhase);
        bool const graphDecoderStep = useDecodeGraph && !contextPhase;
        if (!graphDecoderStep)
        {
            if (!decoderContext->enqueueV3(stream))
            {
                throw std::runtime_error("decoder enqueueV3 failed");
            }
        }

        if (contextPhase)
        {
            tileContextCaches(batchSize, beamWidth, maxEncoderInputLen);
            tileLogits(batchSize, beamWidth);
        }

        auto logitsPtr = contextPhase && beamWidth > 1 ? logitsTiled.ptr : logits.ptr;
        DynamicDecodeAdapter::ForwardParams params;
        params.logits = wrapTensor(logitsPtr, nvinfer1::DataType::kHALF, dims({batchSize, beamWidth, kVocabSizePadded}));
        params.endIds = wrapTensor(decoderScalarInts.ptr, nvinfer1::DataType::kINT32, dims({batchSize}));
        if (refreshEndIds)
        {
            hostEndIds.assign(static_cast<size_t>(batchSize), kEosId);
            copyHostToDevice(decoderScalarInts, hostEndIds.data(), hostEndIds.size() * sizeof(int32_t));
        }
        params.outputIds = wrapTensor(outputIds.ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth, maxSeqLen}));
        params.newTokens = wrapTensor(newTokens.ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth}));
        params.finished = wrapTensor(finished.ptr, nvinfer1::DataType::kUINT8, dims({batchSize, beamWidth}));
        params.sequenceLengths = wrapTensor(sequenceLength.ptr, nvinfer1::DataType::kINT32, dims({batchSize * beamWidth}));
        params.cumLogProbs = wrapTensor(cumLogProbs.ptr, nvinfer1::DataType::kFLOAT, dims({batchSize * beamWidth}));
        params.parentIds = wrapTensor(parentIds.ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth, maxSeqLen}));
        int32_t const cacheSrcIndex = step % 2 == 0 ? 0 : 1;
        int32_t const cacheTgtIndex = step % 2 == 0 ? 1 : 0;
        params.srcCacheIndirection
            = wrapTensor(cacheIndirection[cacheSrcIndex].ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth, maxSeqLen}));
        params.tgtCacheIndirection
            = wrapTensor(cacheIndirection[cacheTgtIndex].ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth, maxSeqLen}));
        if (beamWidth > 1)
        {
            params.beamHypsOutputIdsCba = wrapTensor(
                beamHypsOutputIdsCba.ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth * 2, maxSeqLen}));
            params.beamHypsSeqLenCba
                = wrapTensor(beamHypsSeqLenCba.ptr, nvinfer1::DataType::kINT32, dims({batchSize, beamWidth * 2}));
            params.beamHypsCumLogProbsCba
                = wrapTensor(beamHypsCumLogProbsCba.ptr, nvinfer1::DataType::kFLOAT, dims({batchSize, beamWidth * 2}));
            params.beamHypsNormedScoresCba
                = wrapTensor(beamHypsNormedScoresCba.ptr, nvinfer1::DataType::kFLOAT, dims({batchSize, beamWidth * 2}));
            params.beamHypsLogProbsCba = wrapTensor(
                beamHypsLogProbsCba.ptr, nvinfer1::DataType::kFLOAT, dims({batchSize, beamWidth * 2, maxSeqLen}));
            params.beamHypsMinNormedScores
                = wrapTensor(beamHypsMinNormedScores.ptr, nvinfer1::DataType::kFLOAT, dims({batchSize}));
            params.beamHypsNumBeams = wrapTensor(beamHypsNumBeams.ptr, nvinfer1::DataType::kINT32, dims({batchSize}));
            params.beamHypsIsDone = wrapTensor(beamHypsIsDone.ptr, nvinfer1::DataType::kBOOL, dims({batchSize}));
        }
        params.inputLengths = wrapTensor(contextLengths.ptr, nvinfer1::DataType::kINT32, dims({batchSize * beamWidth}));
        params.sequenceLimitLength
            = wrapTensor(sequenceLimitLength.ptr, nvinfer1::DataType::kINT32, dims({batchSize}));
        params.step = step + 1;
        params.maxInputLength = 1;
        params.maxAttentionWindow = maxSeqLen;
        params.sinkTokenLength = 0;
        params.iteration = 0;
        params.localBatchSize = batchSize;
        params.forceFinishedCheck = forceFinishedCheck;
        params.deferFinishedSync = graphDecoderStep ? true : deferFinishedSync;
        if (graphDecoderStep)
        {
            runDecodeStepGraph(beamWidth, step, params);
            return false;
        }
        return decoder->forward(params);
    }

    DeviceBuffer& finalizeBeamSearch(int32_t batchSize, int32_t beamWidth)
    {
        if (beamWidth == 1)
        {
            return outputIds;
        }

        hostLengthPenalties.assign(batchSize, 1.0f);
        copyHostToDevice(lengthPenalty, hostLengthPenalties.data(), hostLengthPenalties.size() * sizeof(float));
        tk::invokeInitializeOutput(finalizedOutputIds.as<int32_t>(), decoderScalarInts.as<int32_t>(), batchSize,
            beamWidth, maxSeqLen, stream);

        tk::BeamHypotheses bh;
        bh.nMaxBatchSize = batchSize;
        bh.nBatchSize = batchSize;
        bh.nBeamWidth = beamWidth;
        bh.nMaxSeqLen = maxSeqLen;
        bh.lengthPenalties = lengthPenalty.as<float>();
        bh.inputLengths = contextLengths.as<int32_t>();
        bh.outputIds = finalizedOutputIds.as<int32_t>();
        bh.logProbs = nullptr;
        bh.logProbsTiled = nullptr;
        bh.sequenceLengths = sequenceLength.as<int32_t>();
        bh.cumLogProbs = cumLogProbs.as<float>();
        bh.outputIdsCBA = beamHypsOutputIdsCba.as<int32_t>();
        bh.logProbsCBA = nullptr;
        bh.sequenceLengthsCBA = beamHypsSeqLenCba.as<int32_t>();
        bh.cumLogProbsCBA = beamHypsCumLogProbsCba.as<float>();
        bh.normedScoresCBA = beamHypsNormedScoresCba.as<float>();
        bh.numBeamsCBA = beamHypsNumBeams.as<int32_t>();
        bh.minNormedScoresCBA = beamHypsMinNormedScores.as<float>();
        bh.batchDones = reinterpret_cast<bool*>(beamHypsIsDone.ptr);
        bh.finished = reinterpret_cast<tk::FinishedState*>(finished.ptr);
        bh.outputIdsUnfinish = outputIds.as<int32_t>();
        bh.parentIdsUnfinish = parentIds.as<int32_t>();

        tk::invokeInsertUnfinishedPath(bh, stream);
        tk::invokeFinalize(bh, stream);
        return finalizedOutputIds;
    }

    bool canUseDecodeGraph(std::vector<int32_t> const& encoderLengths, int32_t realBatchSize, int32_t beamWidth) const
    {
        if (!cudaGraphDecode || beamWidth != maxBeamWidth || realBatchSize <= 0 || realBatchSize > graphBatchSize)
        {
            return false;
        }
        return std::all_of(encoderLengths.begin(), encoderLengths.end(),
            [this](int32_t len) { return len > 0 && len <= graphEncoderLenBucket; });
    }

    std::string graphKey(int32_t beamWidth, int32_t step) const
    {
        return std::to_string(graphBatchSize) + ":" + std::to_string(beamWidth) + ":"
            + std::to_string(graphEncoderLenBucket) + ":" + std::to_string(step);
    }

    void runDecodeStepGraph(
        int32_t beamWidth, int32_t step, DynamicDecodeAdapter::ForwardParams const& params)
    {
        auto const key = graphKey(beamWidth, step);
        auto it = decodeGraphs.find(key);
        if (params.forceFinishedCheck)
        {
            decoder->clearFinished(params.localBatchSize);
        }
        if (it == decodeGraphs.end())
        {
            DecodeGraph captured;
            checkCuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                "cudaStreamBeginCapture decode step graph");
            if (!decoderContext->enqueueV3(stream))
            {
                throw std::runtime_error("decoder enqueueV3 failed during graph capture");
            }
            decoder->forward(params);
            checkCuda(cudaStreamEndCapture(stream, &captured.graph), "cudaStreamEndCapture decode step graph");
            auto instantiateStatus = cudaGraphInstantiate(&captured.exec, captured.graph, nullptr, nullptr, 0);
            checkCuda(instantiateStatus, "cudaGraphInstantiate decode step graph");
            checkCuda(cudaGraphUpload(captured.exec, stream), "cudaGraphUpload decode step graph");
            it = decodeGraphs.emplace(key, captured).first;
        }

        checkCuda(cudaGraphLaunch(it->second.exec, stream), "cudaGraphLaunch decode step graph");
    }

    bool runDecodeGraphChunks(int32_t realBatchSize, int32_t beamWidth, int32_t totalEncoderTokens,
        int32_t maxEncoderInputLen, int32_t steps)
    {
        bool done = false;
        for (int32_t step = 1; step < steps && !done; ++step)
        {
            bool const forceFinishedCheck = (step % graphChunkSize) == 0 || step == steps - 1;
            (void) runDecodeStep(graphBatchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, step, false,
                false, forceFinishedCheck, false, true);
            if (forceFinishedCheck)
            {
                done = decoder->consumeFinished(realBatchSize);
            }
        }
        return done;
    }

    void warmupDecodeGraphPath(int32_t batchSize, int32_t beamWidth, int32_t totalEncoderTokens,
        int32_t maxEncoderInputLen, int32_t steps)
    {
        if (decodeGraphWarmupDone || steps <= 0)
        {
            return;
        }

        bool done = runDecodeStep(batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, 0, true);
        for (int32_t step = 1; step < steps && !done; ++step)
        {
            bool const forceFinishedCheck = (step % graphChunkSize) == 0 || step == steps - 1;
            done = runDecodeStep(batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, step, false, false,
                forceFinishedCheck, !forceFinishedCheck);
        }
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize decode graph warmup");
        decodeGraphWarmupDone = true;
    }

    std::vector<std::vector<std::vector<int32_t>>> infer(
        std::vector<std::vector<int32_t>> const& encoderInputs, int32_t requestedNewTokens, int32_t requestedBeamWidth)
    {
        int32_t realBatchSize = static_cast<int32_t>(encoderInputs.size());
        int32_t batchSize = realBatchSize;
        int32_t beamWidth = std::min(requestedBeamWidth, maxBeamWidth);
        int32_t steps = std::min(requestedNewTokens, maxNewTokens);
        if (batchSize <= 0 || batchSize > maxBatchSize || beamWidth <= 0)
        {
            throw std::runtime_error("invalid direct runner batch or beam size");
        }

        std::vector<int32_t> encoderLengths;
        encoderLengths.reserve(encoderInputs.size());
        for (auto const& input : encoderInputs)
        {
            encoderLengths.push_back(static_cast<int32_t>(input.size()));
        }
        bool const useDecodeGraph = canUseDecodeGraph(encoderLengths, realBatchSize, beamWidth);
        if (!useDecodeGraph && !decodeGraphs.empty())
        {
            clearDecodeGraphs();
        }
        std::vector<std::vector<int32_t>> paddedEncoderInputs;
        std::vector<std::vector<int32_t>> const* activeEncoderInputs = &encoderInputs;
        if (useDecodeGraph)
        {
            paddedEncoderInputs = encoderInputs;
            paddedEncoderInputs.resize(static_cast<size_t>(graphBatchSize), std::vector<int32_t>{kPadId});
            encoderLengths.resize(static_cast<size_t>(graphBatchSize), 1);
            batchSize = graphBatchSize;
            activeEncoderInputs = &paddedEncoderInputs;
        }
        contextStartIds.assign(static_cast<size_t>(batchSize) * beamWidth, kDecoderStartId);
        currentEncoderLengths = encoderLengths;
        int32_t totalEncoderTokensHint = useDecodeGraph
            ? graphBatchSize * graphEncoderLenBucket
            : std::accumulate(encoderLengths.begin(), encoderLengths.end(), 0);
        int32_t maxEncoderInputLenHint = useDecodeGraph ? graphEncoderLenBucket : *std::max_element(encoderLengths.begin(), encoderLengths.end());
        allocateFor(batchSize, beamWidth, maxEncoderInputLenHint, totalEncoderTokensHint);

        int32_t totalEncoderTokens = 0;
        int32_t maxEncoderInputLen = 0;
        runEncoder(*activeEncoderInputs, totalEncoderTokens, maxEncoderInputLen, useDecodeGraph ? graphEncoderLenBucket : 0);
        if (useDecodeGraph)
        {
            maxEncoderInputLen = graphEncoderLenBucket;
        }
        tileEncoderLengths(encoderLengths, beamWidth);
        setupDecoder(batchSize, beamWidth);

        bool done = false;
        if (useDecodeGraph)
        {
            warmupDecodeGraphPath(batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, steps);
            if (steps > 0)
            {
                done = runDecodeStep(batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, 0, true);
            }
            if (!done)
            {
                done = runDecodeGraphChunks(realBatchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, steps);
            }
        }
        else
        {
            for (int32_t step = 0; step < steps && !done; ++step)
            {
                done = runDecodeStep(
                    batchSize, beamWidth, totalEncoderTokens, maxEncoderInputLen, step, step == 0, step == 0);
            }
        }

        if (std::getenv("INDICXLIT_DEBUG_RAW") != nullptr)
        {
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize before raw output copy");
            std::vector<int32_t> raw(static_cast<size_t>(batchSize) * beamWidth * maxSeqLen);
            checkCuda(cudaMemcpy(raw.data(), outputIds.ptr, raw.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
                "cudaMemcpy raw output ids");
            std::cerr << "RAW_OUTPUT_IDS\n";
            for (int32_t b = 0; b < batchSize; ++b)
            {
                for (int32_t beam = 0; beam < beamWidth; ++beam)
                {
                    auto const base = (static_cast<size_t>(b) * beamWidth + beam) * maxSeqLen;
                    std::cerr << b << ":" << beam;
                    for (int32_t i = 0; i < maxSeqLen; ++i)
                    {
                        std::cerr << " " << raw[base + i];
                    }
                    std::cerr << "\n";
                }
            }
        }

        DeviceBuffer& output = finalizeBeamSearch(batchSize, beamWidth);
        checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize before output copy");
        hostOutput.resize(static_cast<size_t>(batchSize) * beamWidth * maxSeqLen);
        checkCuda(cudaMemcpy(hostOutput.data(), output.ptr, hostOutput.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy output ids");

        std::vector<std::vector<std::vector<int32_t>>> result(realBatchSize);
        for (int32_t b = 0; b < realBatchSize; ++b)
        {
            result[b].resize(beamWidth);
            for (int32_t beam = 0; beam < beamWidth; ++beam)
            {
                auto const base = (static_cast<size_t>(b) * beamWidth + beam) * maxSeqLen;
                result[b][beam].assign(hostOutput.begin() + base, hostOutput.begin() + base + maxSeqLen);
            }
        }
        return result;
    }

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> encoderEngine;
    std::unique_ptr<nvinfer1::ICudaEngine> decoderEngine;
    std::unique_ptr<nvinfer1::IExecutionContext> encoderContext;
    std::unique_ptr<nvinfer1::IExecutionContext> decoderContext;
    cudaStream_t stream{};
    int32_t maxBatchSize{};
    int32_t maxBeamWidth{};
    int32_t maxSeqLen{};
    int32_t maxNewTokens{};
    int32_t maxEncoderLen{};
    std::unique_ptr<DynamicDecodeAdapter> decoder;
    bool decoderSetupValid{false};
    int32_t decoderSetupBatchSize{-1};
    int32_t decoderSetupBeamWidth{-1};
    bool decoderSetupCumLogProbs{false};
    bool generationBindingsValid{false};
    int32_t generationBindingBatchSize{-1};
    int32_t generationBindingBeamWidth{-1};
    int32_t generationBindingMaxEncoderInputLen{-1};
    bool cudaGraphDecode{false};
    int32_t graphBatchSize{256};
    int32_t graphEncoderLenBucket{64};
    int32_t graphChunkSize{3};
    std::map<std::string, DecodeGraph> decodeGraphs;
    bool decodeGraphWarmupDone{false};
    std::vector<int32_t> contextStartIds;
    std::vector<int32_t> currentEncoderLengths;
    std::vector<int32_t> hostFlatIds;
    std::vector<int32_t> hostFlatPositions;
    std::vector<int32_t> hostEncoderLengths;
    std::vector<int32_t> hostMaxLenShape;
    std::vector<int32_t> hostDecodeOnes;
    std::vector<int32_t> hostPositions;
    std::vector<int32_t> hostLastTokenIds;
    std::vector<int32_t> hostSequenceLimits;
    std::vector<float> hostCumLogProbs;
    std::vector<uint8_t> hostDenseMask;
    std::vector<int32_t> hostTiledEncoderLengths;
    std::vector<int32_t> hostEndIds;
    std::vector<float> hostLengthPenalties;
    std::vector<int32_t> hostOutput;

    DeviceBuffer encoderInputIds;
    DeviceBuffer encoderPositionIds;
    DeviceBuffer encoderInputLengths;
    DeviceBuffer encoderMaxInputLength;
    DeviceBuffer encoderOutput;
    DeviceBuffer inputIds;
    void* decoderInputIdsPtr{nullptr};
    DeviceBuffer positionIds;
    DeviceBuffer encoderInputLengthsTiled;
    DeviceBuffer decoderScalarInts;
    DeviceBuffer contextLengths;
    DeviceBuffer sequenceLength;
    DeviceBuffer sequenceLimitLength;
    DeviceBuffer requestTypes;
    DeviceBuffer lastTokenIds;
    DeviceBuffer crossKvCacheGen;
    DeviceBuffer crossAttentionMask;
    DeviceBuffer crossAttentionCuQSeqLens;
    DeviceBuffer crossAttentionCuMaskRows;
    DeviceBuffer crossAttentionPackedMask;
    DeviceBuffer cacheIndirection[2];
    DeviceBuffer logits;
    DeviceBuffer logitsTiled;
    DeviceBuffer outputIds;
    DeviceBuffer newTokens;
    DeviceBuffer finished;
    DeviceBuffer parentIds;
    DeviceBuffer cumLogProbs;
    DeviceBuffer finalizedOutputIds;
    DeviceBuffer lengthPenalty;
    DeviceBuffer beamHypsOutputIdsCba;
    DeviceBuffer beamHypsSeqLenCba;
    DeviceBuffer beamHypsCumLogProbsCba;
    DeviceBuffer beamHypsNormedScoresCba;
    DeviceBuffer beamHypsLogProbsCba;
    DeviceBuffer beamHypsMinNormedScores;
    DeviceBuffer beamHypsNumBeams;
    DeviceBuffer beamHypsIsDone;
    DeviceBuffer kvContext[kNumLayers];
    DeviceBuffer kv[kNumLayers];
    DeviceBuffer crossKvContext[kNumLayers];
    DeviceBuffer crossKv[kNumLayers];
    DeviceBuffer tileLayerPtrs;
    std::array<TileLayerPtrs, kNumLayers> hostTileLayerPtrs{};
    std::vector<int32_t> hostPastKeyValueLengths;
    std::vector<int32_t> hostContextLengths;
    std::vector<int32_t> hostRequestTypes;
    std::vector<int32_t> hostMaxAttentionWindowSizes;
    std::vector<int32_t> hostSinkTokenLength;
    std::vector<int64_t> hostRuntimePerfKnobs;
    std::vector<int64_t> hostContextProgress;
};

DirectEncDecRunner::DirectEncDecRunner(
    fs::path const& engineDir, int32_t maxBatchSize, int32_t maxBeamWidth, int32_t maxNewTokens)
    : mImpl(std::make_unique<Impl>(engineDir, maxBatchSize, maxBeamWidth, maxNewTokens))
{
}

DirectEncDecRunner::~DirectEncDecRunner() = default;

std::vector<std::vector<std::vector<int32_t>>> DirectEncDecRunner::infer(
    std::vector<std::vector<int32_t>> const& encoderInputIds, int32_t maxNewTokens, int32_t beamWidth)
{
    return mImpl->infer(encoderInputIds, maxNewTokens, beamWidth);
}

} // namespace indicxlit
