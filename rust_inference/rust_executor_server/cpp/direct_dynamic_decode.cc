#include "direct_dynamic_decode.h"

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"
#include "tensorrt_llm/layers/decodingParams.h"
#include "tensorrt_llm/layers/dynamicDecodeLayer.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/decodingLayerWorkspace.h"
#include "tensorrt_llm/runtime/gptDecoder.h"

#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tr = tensorrt_llm::runtime;
namespace tl = tensorrt_llm::layers;
namespace tk = tensorrt_llm::kernels;
namespace tle = tensorrt_llm::executor;

namespace indicxlit
{

struct DynamicDecodeAdapter::Impl
{
    Impl(int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize, int32_t vocabSizePadded, cudaStream_t externalStream)
        : maxBatchSize(maxBatchSize)
        , maxBeamWidth(maxBeamWidth)
        , vocabSize(vocabSize)
        , vocabSizePadded(vocabSizePadded)
        , domain(maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded)
    {
        cudaStream_t rawStream = externalStream;
        bool ownsStream = false;
        if (!rawStream)
        {
            cudaStreamCreateWithFlags(&rawStream, cudaStreamNonBlocking);
            ownsStream = true;
        }
        stream = std::make_shared<tr::CudaStream>(rawStream, 0, ownsStream);
        bufferManager = std::make_shared<tr::BufferManager>(stream);
        layer = std::make_shared<tl::DynamicDecodeLayer<half>>(tle::DecodingMode::Auto(), domain, bufferManager);
        batchSlots = tr::getDefaultBatchSlots(maxBatchSize);
        workspace = std::make_shared<tr::DecodingLayerWorkspace>(
            bufferManager, domain, nvinfer1::DataType::kHALF, layer->getWorkspaceSize());
        finishedSum = tr::BufferManager::pinnedPool(
            tr::ITensor::makeShape({maxBatchSize}), nvinfer1::DataType::kINT32);
        disableFinishedSync = std::getenv("INDICXLIT_DISABLE_FINISHED_SYNC") != nullptr;
        if (auto const* interval = std::getenv("INDICXLIT_FINISHED_SYNC_INTERVAL"))
        {
            finishedSyncInterval = std::atoi(interval);
        }
    }

    ~Impl()
    {
        if (stream)
        {
            cudaStreamSynchronize(stream->get());
        }
    }

    int32_t maxBatchSize;
    int32_t maxBeamWidth;
    int32_t vocabSize;
    int32_t vocabSizePadded;
    int32_t beamWidth{1};
    tl::DecoderDomain domain;
    std::shared_ptr<tr::CudaStream> stream;
    std::shared_ptr<tr::BufferManager> bufferManager;
    std::shared_ptr<tl::DynamicDecodeLayer<half>> layer;
    std::shared_ptr<tr::DecodingLayerWorkspace> workspace;
    tr::ITensor::SharedConstPtr batchSlots;
    tr::ITensor::SharedPtr finishedSum;
    bool disableFinishedSync{false};
    int32_t finishedSyncInterval{3};
};

DynamicDecodeAdapter::DynamicDecodeAdapter(
    int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize, int32_t vocabSizePadded)
    : mImpl(std::make_unique<Impl>(maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, nullptr))
{
}

DynamicDecodeAdapter::DynamicDecodeAdapter(
    int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize, int32_t vocabSizePadded, cudaStream_t stream)
    : mImpl(std::make_unique<Impl>(maxBatchSize, maxBeamWidth, vocabSize, vocabSizePadded, stream))
{
}

DynamicDecodeAdapter::~DynamicDecodeAdapter() = default;

void DynamicDecodeAdapter::setup(int32_t batchSize, int32_t beamWidth, bool outputLogProbs, bool cumLogProbs)
{
    mImpl->beamWidth = beamWidth;
    auto setupParams = std::make_shared<tl::DynamicDecodeSetupParams>();
    auto penaltyParams = std::make_shared<tl::PenaltySetupParams>();
    auto banWordsParams = std::make_shared<tl::BanWordsSetupParams>();

    if (beamWidth == 1)
    {
        auto decodingParams = std::make_shared<tl::SamplingSetupParams>();
        decodingParams->runtimeTopK = std::vector<tr::SizeType32>(batchSize, 1);
        decodingParams->outputLogProbs = std::vector<bool>(batchSize, outputLogProbs);
        decodingParams->cumLogProbs = std::vector<bool>(batchSize, cumLogProbs);
        setupParams->decodingParams = decodingParams;
    }
    else
    {
        auto decodingParams = std::make_shared<tl::BeamSearchSetupParams>();
        decodingParams->beamSearchDiversityRate = std::vector<float>(batchSize, 0.0f);
        decodingParams->lengthPenalty = std::vector<float>(batchSize, 1.0f);
        decodingParams->earlyStopping = std::vector<int>(batchSize, 1);
        decodingParams->outputLogProbs = std::vector<bool>(batchSize, outputLogProbs);
        decodingParams->cumLogProbs = std::vector<bool>(batchSize, cumLogProbs);
        setupParams->decodingParams = decodingParams;
    }

    setupParams->penaltyParams = penaltyParams;
    setupParams->banWordsParams = banWordsParams;
    tl::TensorConstPtr batchSlotsSlice{tr::ITensor::slice(mImpl->batchSlots, 0, batchSize)};
    mImpl->layer->setup(batchSize, beamWidth, batchSlotsSlice, setupParams, mImpl->workspace);
}

bool DynamicDecodeAdapter::forward(ForwardParams const& p)
{
    if (!p.logits || !p.endIds || !p.outputIds || !p.newTokens)
    {
        throw std::runtime_error("DynamicDecodeAdapter::forward missing mandatory tensors");
    }

    auto const localBatchSize = p.localBatchSize > 0 ? p.localBatchSize : p.logits->getShape().d[0];
    auto const isBeamSearch = mImpl->beamWidth > 1;
    tl::TensorConstPtr batchSlotsSlice{tr::ITensor::slice(mImpl->batchSlots, 0, localBatchSize)};

    std::shared_ptr<tl::DecodingInputs> inputs;
    if (isBeamSearch)
    {
        inputs = std::make_shared<tl::DecodingInputs>(p.endIds, batchSlotsSlice, p.step,
            static_cast<tr::SizeType32>(p.iteration), localBatchSize, p.maxAttentionWindow, p.sinkTokenLength);
    }
    else
    {
        inputs = std::make_shared<tl::SamplingInputs>(
            p.endIds, batchSlotsSlice, p.step, static_cast<tr::SizeType32>(p.iteration), localBatchSize);
    }
    inputs->logits = p.logits;
    inputs->stopCriteriaInputs = std::make_shared<tl::StopCriteriaDecodingInputs>(localBatchSize);
    inputs->banWordsInputs = std::make_shared<tl::BanWordsDecodingInputs>(localBatchSize);
    if (p.inputLengths)
    {
        inputs->inputLengths = p.inputLengths;
    }
    if (p.sequenceLimitLength)
    {
        inputs->stopCriteriaInputs->sequenceLimitLength = p.sequenceLimitLength;
    }
    if (p.finished)
    {
        inputs->finished = p.finished;
    }
    if (p.srcCacheIndirection)
    {
        inputs->srcCacheIndirection = p.srcCacheIndirection;
    }
    if (isBeamSearch)
    {
        inputs->beamSearchSteps = std::vector<tr::SizeType32>(localBatchSize, p.step);
    }

    std::shared_ptr<tl::BaseDecodingOutputs> outputs;
    if (isBeamSearch)
    {
        auto beamOutputs = std::make_shared<tl::BeamSearchOutputs>(p.outputIds);
        if (!p.tgtCacheIndirection)
        {
            throw std::runtime_error("beam search requires target cache indirection");
        }
        beamOutputs->tgtCacheIndirection = p.tgtCacheIndirection;
        if (!p.beamHypsOutputIdsCba || !p.beamHypsSeqLenCba || !p.beamHypsCumLogProbsCba
            || !p.beamHypsNormedScoresCba || !p.beamHypsLogProbsCba || !p.beamHypsMinNormedScores
            || !p.beamHypsNumBeams || !p.beamHypsIsDone)
        {
            throw std::runtime_error("beam search requires BeamHypotheses buffers");
        }
        beamOutputs->beamHypotheses = std::make_unique<tk::BeamHypotheses>();
        beamOutputs->beamHypotheses->outputIdsCBA = tr::bufferCast<tr::TokenIdType>(*p.beamHypsOutputIdsCba);
        beamOutputs->beamHypotheses->sequenceLengthsCBA = tr::bufferCast<tr::SizeType32>(*p.beamHypsSeqLenCba);
        beamOutputs->beamHypotheses->cumLogProbsCBA = tr::bufferCast<float>(*p.beamHypsCumLogProbsCba);
        beamOutputs->beamHypotheses->normedScoresCBA = tr::bufferCast<float>(*p.beamHypsNormedScoresCba);
        beamOutputs->beamHypotheses->logProbsCBA = tr::bufferCast<float>(*p.beamHypsLogProbsCba);
        beamOutputs->beamHypotheses->minNormedScoresCBA = tr::bufferCast<float>(*p.beamHypsMinNormedScores);
        beamOutputs->beamHypotheses->numBeamsCBA = tr::bufferCast<tr::SizeType32>(*p.beamHypsNumBeams);
        beamOutputs->beamHypotheses->batchDones = tr::bufferCast<bool>(*p.beamHypsIsDone);
        outputs = beamOutputs;
    }
    else
    {
        outputs = std::make_shared<tl::BaseDecodingOutputs>(p.outputIds);
    }

    outputs->newTokens = p.newTokens;
    if (p.finished)
    {
        outputs->finished = p.finished;
    }
    if (p.sequenceLengths)
    {
        outputs->sequenceLength = p.sequenceLengths;
    }
    if (p.cumLogProbs)
    {
        outputs->cumLogProbs = p.cumLogProbs;
    }
    if (p.parentIds)
    {
        outputs->parentIds = p.parentIds;
    }

    tr::SizeType32* finishedSumHost = nullptr;
    bool const checkFinished = !mImpl->disableFinishedSync
        && (p.forceFinishedCheck || (mImpl->finishedSyncInterval > 0
                                      && ((p.step + 1) % mImpl->finishedSyncInterval == 0)));
    if (checkFinished && p.sequenceLimitLength && p.finished)
    {
        outputs->finishedSum = mImpl->finishedSum;
        finishedSumHost = tr::bufferCast<tr::SizeType32>(*mImpl->finishedSum);
        for (int32_t bi = 0; bi < localBatchSize; ++bi)
        {
            finishedSumHost[bi] = 0;
        }
    }

    mImpl->layer->forwardAsync(outputs, inputs, mImpl->workspace);
    if (p.deferFinishedSync)
    {
        return false;
    }
    if (!checkFinished)
    {
        return false;
    }
    cudaStreamSynchronize(mImpl->stream->get());

    if (!finishedSumHost)
    {
        return false;
    }
    uint32_t finishedCount = 0;
    for (int32_t bi = 0; bi < localBatchSize; ++bi)
    {
        finishedCount += finishedSumHost[bi];
    }
    return finishedCount == static_cast<uint32_t>(localBatchSize * mImpl->beamWidth);
}

bool DynamicDecodeAdapter::consumeFinished(int32_t localBatchSize)
{
    cudaStreamSynchronize(mImpl->stream->get());
    auto* finishedSumHost = tr::bufferCast<tr::SizeType32>(*mImpl->finishedSum);
    uint32_t finishedCount = 0;
    for (int32_t bi = 0; bi < localBatchSize; ++bi)
    {
        finishedCount += finishedSumHost[bi];
    }
    return finishedCount == static_cast<uint32_t>(localBatchSize * mImpl->beamWidth);
}

void DynamicDecodeAdapter::clearFinished(int32_t localBatchSize)
{
    auto* finishedSumHost = tr::bufferCast<tr::SizeType32>(*mImpl->finishedSum);
    for (int32_t bi = 0; bi < localBatchSize; ++bi)
    {
        finishedSumHost[bi] = 0;
    }
}

} // namespace indicxlit
