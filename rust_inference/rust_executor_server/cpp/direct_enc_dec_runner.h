#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace indicxlit
{

class DirectEncDecRunner
{
public:
    DirectEncDecRunner(std::filesystem::path const& engineDir, int32_t maxBatchSize, int32_t maxBeamWidth,
        int32_t maxNewTokens);
    ~DirectEncDecRunner();

    std::vector<std::vector<std::vector<int32_t>>> infer(
        std::vector<std::vector<int32_t>> const& encoderInputIds, int32_t maxNewTokens, int32_t beamWidth);

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};

} // namespace indicxlit
