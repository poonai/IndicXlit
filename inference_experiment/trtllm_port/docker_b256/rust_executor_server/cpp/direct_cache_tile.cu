#include "direct_cache_tile.h"

namespace indicxlit
{
namespace
{

__device__ __forceinline__ void copyHalfRowVectorized(half const* __restrict__ src, half* __restrict__ dst,
    size_t elems)
{
    constexpr size_t vecHalfCount = sizeof(uint4) / sizeof(half);
    size_t prefix = 0;
    auto const srcAddr = reinterpret_cast<uintptr_t>(src);
    auto const dstAddr = reinterpret_cast<uintptr_t>(dst);
    while (prefix < elems
        && (((srcAddr + prefix * sizeof(half)) & (sizeof(uint4) - 1)) != 0
               || ((dstAddr + prefix * sizeof(half)) & (sizeof(uint4) - 1)) != 0))
    {
        dst[prefix] = src[prefix];
        ++prefix;
    }

    auto const vecElems = (elems - prefix) / vecHalfCount;
    auto const* srcVec = reinterpret_cast<uint4 const*>(src + prefix);
    auto* dstVec = reinterpret_cast<uint4*>(dst + prefix);
    for (size_t i = 0; i < vecElems; ++i)
    {
        dstVec[i] = srcVec[i];
    }

    auto const tailOffset = prefix + vecElems * vecHalfCount;
    for (size_t i = tailOffset; i < elems; ++i)
    {
        dst[i] = src[i];
    }
}

__global__ void tileBeamCopyHalfKernel(
    half const* __restrict__ src, half* __restrict__ dst, int32_t beamWidth, size_t rowElems, size_t totalElems)
{
    auto const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const rows = totalElems / rowElems;
    for (size_t row = idx; row < rows; row += stride)
    {
        auto const srcRow = row / static_cast<size_t>(beamWidth);
        copyHalfRowVectorized(src + srcRow * rowElems, dst + row * rowElems, rowElems);
    }
}

__global__ void tileBeamCopyPairHalfKernel(half const* __restrict__ srcA, half* __restrict__ dstA, size_t rowElemsA,
    size_t totalElemsA, half const* __restrict__ srcB, half* __restrict__ dstB, size_t rowElemsB, size_t totalElemsB,
    int32_t beamWidth)
{
    auto const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const rows = totalElemsA / rowElemsA;
    for (size_t row = idx; row < rows; row += stride)
    {
        auto const srcRow = row / static_cast<size_t>(beamWidth);
        copyHalfRowVectorized(srcA + srcRow * rowElemsA, dstA + row * rowElemsA, rowElemsA);
        copyHalfRowVectorized(srcB + srcRow * rowElemsB, dstB + row * rowElemsB, rowElemsB);
    }
}

__global__ void tileBeamCopyAllLayersKernel(TileLayerPtrs const* __restrict__ layers, int32_t numLayers,
    size_t rowElemsSelf, size_t rowElemsCross, size_t rows, int32_t beamWidth)
{
    auto const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const totalRows = static_cast<size_t>(numLayers) * rows;
    for (size_t linearRow = idx; linearRow < totalRows; linearRow += stride)
    {
        auto const layer = static_cast<int32_t>(linearRow / rows);
        auto const rowInBatch = linearRow - static_cast<size_t>(layer) * rows;
        auto const srcRow = rowInBatch / static_cast<size_t>(beamWidth);
        auto const ptrs = layers[layer];
        copyHalfRowVectorized(ptrs.selfSrc + srcRow * rowElemsSelf, ptrs.selfDst + rowInBatch * rowElemsSelf,
            rowElemsSelf);
        copyHalfRowVectorized(ptrs.crossSrc + srcRow * rowElemsCross, ptrs.crossDst + rowInBatch * rowElemsCross,
            rowElemsCross);
    }
}

__global__ void prepareDecodeInputsKernel(int32_t* positionIds, int32_t* requestTypes, int32_t* lastTokenIds,
    int32_t* contextLengths, int32_t* sequenceLength, int32_t* sequenceLimitLength, uint8_t* crossKvCacheGen,
    uint8_t* __restrict__ mask, int32_t const* __restrict__ lengthsPerRow, int32_t activeRows, int32_t decodeRows,
    int32_t batchSize, int32_t maskCols, int32_t position, int32_t requestType, int32_t maxSeqLen, int32_t initSequence)
{
    auto const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    if (idx == 0)
    {
        *crossKvCacheGen = requestType == 0 ? 1 : 0;
    }
    for (int32_t row = static_cast<int32_t>(idx); row < decodeRows; row += static_cast<int32_t>(stride))
    {
        contextLengths[row] = 1;
        if (initSequence)
        {
            sequenceLength[row] = 1;
        }
    }
    for (int32_t row = static_cast<int32_t>(idx); row < activeRows; row += static_cast<int32_t>(stride))
    {
        positionIds[row] = position;
        requestTypes[row] = requestType;
        lastTokenIds[row] = row + 1;
    }
    if (initSequence)
    {
        for (int32_t row = static_cast<int32_t>(idx); row < batchSize; row += static_cast<int32_t>(stride))
        {
            sequenceLimitLength[row] = maxSeqLen;
        }
    }
    auto const maskElems = static_cast<size_t>(activeRows) * static_cast<size_t>(maskCols);
    for (size_t linear = idx; linear < maskElems; linear += stride)
    {
        auto const row = static_cast<int32_t>(linear / static_cast<size_t>(maskCols));
        auto const col = static_cast<int32_t>(linear - static_cast<size_t>(row) * static_cast<size_t>(maskCols));
        auto const valid = lengthsPerRow[row] < maskCols ? lengthsPerRow[row] : maskCols;
        mask[linear] = col < valid ? 1 : 0;
    }
}

__global__ void buildPrefixMaskKernel(
    uint8_t* __restrict__ mask, int32_t const* __restrict__ lengthsPerRow, int32_t rows, int32_t maskCols)
{
    auto const total = static_cast<size_t>(rows) * static_cast<size_t>(maskCols);
    auto const idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t linear = idx; linear < total; linear += stride)
    {
        auto const row = static_cast<int32_t>(linear / static_cast<size_t>(maskCols));
        auto const col = static_cast<int32_t>(linear - static_cast<size_t>(row) * static_cast<size_t>(maskCols));
        auto const valid = lengthsPerRow[row] < maskCols ? lengthsPerRow[row] : maskCols;
        mask[linear] = col < valid ? 1 : 0;
    }
}

} // namespace

cudaError_t launchTileBeamCopyHalf(
    half const* src, half* dst, int32_t batchSize, int32_t beamWidth, size_t rowElems, cudaStream_t stream)
{
    if (batchSize <= 0 || beamWidth <= 0 || rowElems == 0)
    {
        return cudaSuccess;
    }
    auto const totalElems = static_cast<size_t>(batchSize) * static_cast<size_t>(beamWidth) * rowElems;
    int constexpr blockSize = 256;
    auto blocks = static_cast<int>((totalElems + blockSize - 1) / blockSize);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > 4096 ? 4096 : blocks;
    tileBeamCopyHalfKernel<<<blocks, blockSize, 0, stream>>>(src, dst, beamWidth, rowElems, totalElems);
    return cudaGetLastError();
}

cudaError_t launchTileBeamCopyPairHalf(half const* srcA, half* dstA, size_t rowElemsA, half const* srcB, half* dstB,
    size_t rowElemsB, int32_t batchSize, int32_t beamWidth, cudaStream_t stream)
{
    if (batchSize <= 0 || beamWidth <= 0 || rowElemsA == 0 || rowElemsB == 0)
    {
        return cudaSuccess;
    }
    auto const rows = static_cast<size_t>(batchSize) * static_cast<size_t>(beamWidth);
    auto const totalElemsA = rows * rowElemsA;
    auto const totalElemsB = rows * rowElemsB;
    int constexpr blockSize = 256;
    auto blocks = static_cast<int>((totalElemsA + totalElemsB + blockSize - 1) / blockSize);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > 4096 ? 4096 : blocks;
    tileBeamCopyPairHalfKernel<<<blocks, blockSize, 0, stream>>>(
        srcA, dstA, rowElemsA, totalElemsA, srcB, dstB, rowElemsB, totalElemsB, beamWidth);
    return cudaGetLastError();
}

cudaError_t launchTileBeamCopyAllLayers(TileLayerPtrs const* layers, int32_t numLayers, size_t rowElemsSelf,
    size_t rowElemsCross, int32_t batchSize, int32_t beamWidth, cudaStream_t stream)
{
    if (!layers || numLayers <= 0 || batchSize <= 0 || beamWidth <= 0 || rowElemsSelf == 0 || rowElemsCross == 0)
    {
        return cudaSuccess;
    }
    auto constexpr vecBytes = sizeof(uint4);
    auto const selfBytes = rowElemsSelf * sizeof(half);
    auto const crossBytes = rowElemsCross * sizeof(half);
    if ((selfBytes % vecBytes) != 0 || (crossBytes % vecBytes) != 0)
    {
        return cudaErrorInvalidValue;
    }
    auto const rows = static_cast<size_t>(batchSize) * static_cast<size_t>(beamWidth);
    auto const rowVecsSelf = selfBytes / vecBytes;
    auto const rowVecsCross = crossBytes / vecBytes;
    auto const totalVecs = static_cast<size_t>(numLayers) * rows * (rowVecsSelf + rowVecsCross);
    int constexpr blockSize = 256;
    auto blocks = static_cast<int>((totalVecs + blockSize - 1) / blockSize);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > 4096 ? 4096 : blocks;
    tileBeamCopyAllLayersKernel<<<blocks, blockSize, 0, stream>>>(
        layers, numLayers, rowElemsSelf, rowElemsCross, rows, beamWidth);
    return cudaGetLastError();
}

cudaError_t launchPrepareDecodeInputs(int32_t* positionIds, int32_t* requestTypes, int32_t* lastTokenIds,
    int32_t* contextLengths, int32_t* sequenceLength, int32_t* sequenceLimitLength, uint8_t* crossKvCacheGen,
    uint8_t* mask, int32_t const* lengthsPerRow, int32_t activeRows, int32_t decodeRows, int32_t batchSize,
    int32_t maskCols, int32_t position, int32_t requestType, int32_t maxSeqLen, bool initSequence, cudaStream_t stream)
{
    int constexpr blockSize = 256;
    auto const maxRows = activeRows > decodeRows ? activeRows : decodeRows;
    auto const maskElems = static_cast<size_t>(activeRows) * static_cast<size_t>(maskCols);
    auto const workItems = maskElems > static_cast<size_t>(maxRows) ? maskElems : static_cast<size_t>(maxRows);
    auto blocks = static_cast<int>((workItems + blockSize - 1) / blockSize);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > 4096 ? 4096 : blocks;
    prepareDecodeInputsKernel<<<blocks, blockSize, 0, stream>>>(positionIds, requestTypes, lastTokenIds, contextLengths,
        sequenceLength, sequenceLimitLength, crossKvCacheGen, mask, lengthsPerRow, activeRows, decodeRows, batchSize,
        maskCols, position, requestType, maxSeqLen, initSequence ? 1 : 0);
    return cudaGetLastError();
}

cudaError_t launchBuildPrefixMask(
    uint8_t* mask, int32_t const* lengthsPerRow, int32_t rows, int32_t maskCols, cudaStream_t stream)
{
    if (rows <= 0 || maskCols <= 0)
    {
        return cudaSuccess;
    }
    auto const total = static_cast<size_t>(rows) * static_cast<size_t>(maskCols);
    int constexpr blockSize = 256;
    auto blocks = static_cast<int>((total + blockSize - 1) / blockSize);
    blocks = blocks < 1 ? 1 : blocks;
    blocks = blocks > 4096 ? 4096 : blocks;
    buildPrefixMaskKernel<<<blocks, blockSize, 0, stream>>>(mask, lengthsPerRow, rows, maskCols);
    return cudaGetLastError();
}

} // namespace indicxlit
