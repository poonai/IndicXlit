#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace indicxlit
{

struct TileLayerPtrs
{
    half const* selfSrc;
    half* selfDst;
    half const* crossSrc;
    half* crossDst;
};

cudaError_t launchTileBeamCopyHalf(
    half const* src, half* dst, int32_t batchSize, int32_t beamWidth, size_t rowElems, cudaStream_t stream);

cudaError_t launchTileBeamCopyPairHalf(half const* srcA, half* dstA, size_t rowElemsA, half const* srcB, half* dstB,
    size_t rowElemsB, int32_t batchSize, int32_t beamWidth, cudaStream_t stream);

cudaError_t launchTileBeamCopyAllLayers(TileLayerPtrs const* layers, int32_t numLayers, size_t rowElemsSelf,
    size_t rowElemsCross, int32_t batchSize, int32_t beamWidth, cudaStream_t stream);

cudaError_t launchPrepareDecodeInputs(int32_t* positionIds, int32_t* requestTypes, int32_t* lastTokenIds,
    int32_t* contextLengths, int32_t* sequenceLength, int32_t* sequenceLimitLength, uint8_t* crossKvCacheGen,
    uint8_t* mask, int32_t const* lengthsPerRow, int32_t activeRows, int32_t decodeRows, int32_t batchSize,
    int32_t maskCols, int32_t position, int32_t requestType, int32_t maxSeqLen, bool initSequence, cudaStream_t stream);

cudaError_t launchBuildPrefixMask(
    uint8_t* mask, int32_t const* lengthsPerRow, int32_t rows, int32_t maskCols, cudaStream_t stream);

} // namespace indicxlit
