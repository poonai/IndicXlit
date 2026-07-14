#pragma once

#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <optional>

namespace indicxlit
{

class DynamicDecodeAdapter
{
public:
    DynamicDecodeAdapter(int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize, int32_t vocabSizePadded);
    DynamicDecodeAdapter(
        int32_t maxBatchSize, int32_t maxBeamWidth, int32_t vocabSize, int32_t vocabSizePadded, cudaStream_t stream);
    ~DynamicDecodeAdapter();

    void setup(int32_t batchSize, int32_t beamWidth, bool outputLogProbs = false, bool cumLogProbs = false);

    struct ForwardParams
    {
        tensorrt_llm::runtime::ITensor::SharedConstPtr logits; // [batch, beam, vocab_padded], gpu, half
        tensorrt_llm::runtime::ITensor::SharedConstPtr endIds; // [batch * beam], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr outputIds;   // [batch, beam, max_seq_len], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr newTokens;   // [batch, beam], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr finished;    // [batch, beam], gpu, uint8
        tensorrt_llm::runtime::ITensor::SharedPtr sequenceLengths; // [batch * beam], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr cumLogProbs;     // [batch * beam], gpu, float, optional
        tensorrt_llm::runtime::ITensor::SharedPtr parentIds;       // [batch, beam, max_seq_len], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr srcCacheIndirection; // [batch, beam, max_seq_len], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr tgtCacheIndirection; // [batch, beam, max_seq_len], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsOutputIdsCba; // [batch, beam * 2, max_seq_len], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsSeqLenCba; // [batch, beam * 2], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsCumLogProbsCba; // [batch, beam * 2], gpu, float
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsNormedScoresCba; // [batch, beam * 2], gpu, float
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsLogProbsCba; // [batch, beam * 2, max_seq_len], gpu, float
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsMinNormedScores; // [batch], gpu, float
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsNumBeams; // [batch], gpu, int32
        tensorrt_llm::runtime::ITensor::SharedPtr beamHypsIsDone; // [batch], gpu, bool
        tensorrt_llm::runtime::ITensor::SharedConstPtr inputLengths; // [batch * beam], gpu, int32, optional
        tensorrt_llm::runtime::ITensor::SharedConstPtr sequenceLimitLength; // [batch], gpu, int32, optional
        int32_t step{0};
        int32_t maxInputLength{1};
        int32_t maxAttentionWindow{0};
        int32_t sinkTokenLength{0};
        uint32_t iteration{0};
        int32_t localBatchSize{0};
        bool forceFinishedCheck{false};
        bool deferFinishedSync{false};
    };

    // Returns true when all beams have finished according to the stop-criteria layer.
    bool forward(ForwardParams const& params);
    void clearFinished(int32_t localBatchSize);
    bool consumeFinished(int32_t localBatchSize);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace indicxlit
