#include "indicxlit_bridge.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
void check(int code, char* error)
{
    if (code != 0)
    {
        std::string message = error ? error : "unknown error";
        if (error)
        {
            indicxlit_free_string(error);
        }
        throw std::runtime_error(message);
    }
}
}

int main(int argc, char** argv)
{
    std::vector<std::string> words;
    for (int i = 1; i < argc; ++i)
    {
        words.emplace_back(argv[i]);
    }
    if (words.empty())
    {
        words = {"namaste", "bharat", "kiran"};
    }

    IndicXlitConfig config{};
    std::string engineDir = std::getenv("ENGINE_DIR") ? std::getenv("ENGINE_DIR") : "/models/engines/b256_cont";
    std::string assetRoot
        = std::getenv("INDICXLIT_MODEL_ROOT") ? std::getenv("INDICXLIT_MODEL_ROOT") : "/models/assets/en2indic";
    config.engine_dir = engineDir.c_str();
    config.asset_root = assetRoot.c_str();
    config.max_batch_size = std::getenv("INDICXLIT_MAX_BATCH_SIZE") ? std::atoi(std::getenv("INDICXLIT_MAX_BATCH_SIZE")) : 256;
    config.max_beam_width = 5;
    config.max_num_tokens = 32;
    config.use_static_scheduler = 1;
    int32_t beamWidth = std::getenv("INDICXLIT_BEAM_WIDTH") ? std::atoi(std::getenv("INDICXLIT_BEAM_WIDTH")) : 5;
    int32_t topk = std::getenv("INDICXLIT_TOPK") ? std::atoi(std::getenv("INDICXLIT_TOPK")) : std::min(beamWidth, 3);

    char* error = nullptr;
    IndicXlitHandle* handle = indicxlit_create(&config, &error);
    if (!handle)
    {
        check(1, error);
    }

    std::vector<char const*> ptrs;
    ptrs.reserve(words.size());
    for (auto const& word : words)
    {
        ptrs.push_back(word.c_str());
    }

    char** outputs = nullptr;
    size_t outputCount = 0;
    error = nullptr;
    int32_t code = indicxlit_infer_batch(handle, ptrs.data(), ptrs.size(), "hi", 32, beamWidth, topk, &outputs, &outputCount, &error);
    check(code, error);
    for (size_t i = 0; i < outputCount; ++i)
    {
        std::cout << words.at(i) << "\t" << outputs[i] << "\n";
    }
    indicxlit_free_string_array(outputs, outputCount);
    indicxlit_destroy(handle);
    return 0;
}
