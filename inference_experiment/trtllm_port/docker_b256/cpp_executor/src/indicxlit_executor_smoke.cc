#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/tensor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace texec = tensorrt_llm::executor;
namespace fs = std::filesystem;

namespace
{
constexpr int32_t kBosId = 0;
constexpr int32_t kPadId = 1;
constexpr int32_t kEosId = 2;
constexpr int32_t kUnkId = 3;

std::vector<std::string> readBaseSymbols(fs::path const& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("failed to open dictionary: " + path.string());
    }

    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }
        auto const pos = line.find_last_of(' ');
        if (pos != std::string::npos && pos > 0)
        {
            out.push_back(line.substr(0, pos));
        }
    }
    return out;
}

std::vector<std::string> readLanguageTokens(fs::path const& path)
{
    std::ifstream in(path);
    if (!in)
    {
        return {};
    }

    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line))
    {
        if (!line.empty())
        {
            out.push_back("__" + line + "__");
        }
    }
    return out;
}

std::vector<std::string> makeSymbols(fs::path const& dictPath, fs::path const& langListPath)
{
    std::vector<std::string> symbols{"<s>", "<pad>", "</s>", "<unk>"};
    auto base = readBaseSymbols(dictPath);
    symbols.insert(symbols.end(), base.begin(), base.end());
    auto langs = readLanguageTokens(langListPath);
    symbols.insert(symbols.end(), langs.begin(), langs.end());
    return symbols;
}

std::unordered_map<std::string, int32_t> makeTokenToId(std::vector<std::string> const& symbols)
{
    std::unordered_map<std::string, int32_t> tokenToId;
    tokenToId.reserve(symbols.size());
    for (size_t i = 0; i < symbols.size(); ++i)
    {
        tokenToId.emplace(symbols[i], static_cast<int32_t>(i));
    }
    return tokenToId;
}

std::string lowerAscii(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

texec::VecTokens encodeWord(
    std::string const& word, std::string const& targetLang, std::unordered_map<std::string, int32_t> const& tokenToId)
{
    texec::VecTokens ids;
    auto const langToken = "__" + targetLang + "__";
    auto const langIt = tokenToId.find(langToken);
    ids.push_back(langIt != tokenToId.end() ? langIt->second : kUnkId);

    auto lowered = lowerAscii(word);
    for (char ch : lowered)
    {
        std::string piece(1, ch);
        auto const pieceIt = tokenToId.find(piece);
        ids.push_back(pieceIt != tokenToId.end() ? pieceIt->second : kUnkId);
    }
    ids.push_back(kEosId);
    return ids;
}

texec::Tensor makeCrossAttentionMask(int32_t rows, int32_t encoderLen)
{
    auto mask = texec::Tensor::cpu<bool>({rows, encoderLen});
    auto* data = static_cast<bool*>(mask.getData());
    std::fill(data, data + static_cast<size_t>(rows) * static_cast<size_t>(encoderLen), true);
    return mask;
}

std::string decode(texec::VecTokens const& tokens, std::vector<std::string> const& targetSymbols)
{
    std::ostringstream spaced;
    bool first = true;
    for (auto token : tokens)
    {
        if (token == kEosId)
        {
            break;
        }
        if (token == kBosId || token == kPadId)
        {
            continue;
        }
        if (!first)
        {
            spaced << ' ';
        }
        first = false;
        if (token >= 0 && static_cast<size_t>(token) < targetSymbols.size())
        {
            spaced << targetSymbols[static_cast<size_t>(token)];
        }
        else
        {
            spaced << "<unk>";
        }
    }

    auto withSpaces = spaced.str();
    std::string compact;
    compact.reserve(withSpaces.size());
    for (char ch : withSpaces)
    {
        if (ch != ' ')
        {
            compact.push_back(ch);
        }
    }
    return compact;
}

std::string argValue(int argc, char** argv, std::string const& key, std::string fallback)
{
    for (int i = 1; i + 1 < argc; ++i)
    {
        if (argv[i] == key)
        {
            return argv[i + 1];
        }
    }
    return fallback;
}

int intArg(int argc, char** argv, std::string const& key, int fallback)
{
    return std::stoi(argValue(argc, argv, key, std::to_string(fallback)));
}

void usage(char const* argv0)
{
    std::cerr << "Usage: " << argv0
              << " --engine-dir PATH --asset-root PATH [--lang hi] [--word bharat]"
                 " [--beam-width 5] [--max-tokens 32] [--max-batch-size 256]\n";
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1)
        {
            usage(argv[0]);
            return 2;
        }

        auto const engineDir = fs::path(argValue(argc, argv, "--engine-dir", ""));
        auto const assetRoot = fs::path(argValue(argc, argv, "--asset-root", ""));
        auto const lang = argValue(argc, argv, "--lang", "hi");
        auto const word = argValue(argc, argv, "--word", "bharat");
        auto const beamWidth = intArg(argc, argv, "--beam-width", 5);
        auto const maxTokens = intArg(argc, argv, "--max-tokens", 32);
        auto const maxBatchSize = intArg(argc, argv, "--max-batch-size", 256);
        auto const maxNumTokens = intArg(argc, argv, "--max-num-tokens", maxBatchSize * std::max(1, maxTokens));
        auto const batching = argValue(argc, argv, "--batching", "inflight");

        if (engineDir.empty() || assetRoot.empty())
        {
            usage(argv[0]);
            return 2;
        }

        auto const langList = assetRoot / "lang_list.txt";
        auto sourceSymbols = makeSymbols(assetRoot / "corpus-bin" / "dict.en.txt", langList);
        auto targetSymbols = makeSymbols(assetRoot / "corpus-bin" / ("dict." + lang + ".txt"), langList);
        auto tokenToId = makeTokenToId(sourceSymbols);

        if (!initTrtLlmPlugins())
        {
            throw std::runtime_error("failed to initialize TensorRT-LLM plugins");
        }

        texec::DynamicBatchConfig dynamicBatchConfig(true, true);
        auto const schedulerPolicy = batching == "static-scheduler" ? texec::CapacitySchedulerPolicy::kSTATIC_BATCH
                                                                     : texec::CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT;
        texec::SchedulerConfig schedulerConfig(schedulerPolicy, std::nullopt, dynamicBatchConfig);
        texec::KvCacheConfig kvCacheConfig(false);
        texec::ExecutorConfig executorConfig(beamWidth, schedulerConfig, kvCacheConfig, true, true);
        executorConfig.setBatchingType(batching == "static" ? texec::BatchingType::kSTATIC : texec::BatchingType::kINFLIGHT);
        executorConfig.setMaxBatchSize(maxBatchSize);
        executorConfig.setMaxNumTokens(maxNumTokens);

        std::cerr << "[indicxlit-cpp] loading executor from " << engineDir << "\n";
        texec::Executor executor(engineDir / "encoder", engineDir / "decoder", texec::ModelType::kENCODER_DECODER, executorConfig);

        auto encoderIds = encodeWord(word, lang, tokenToId);
        texec::VecTokens decoderInput{kEosId};
        auto mask = makeCrossAttentionMask(std::max(1, maxTokens + 1), static_cast<int32_t>(encoderIds.size()));
        texec::SamplingConfig samplingConfig(beamWidth);
        texec::OutputConfig outputConfig(false, false, false, false);

        texec::Request request(decoderInput, maxTokens, false, samplingConfig, outputConfig, kEosId, kPadId,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            encoderIds, std::nullopt, false, texec::Request::kDefaultPriority,
            texec::RequestType::REQUEST_TYPE_CONTEXT_AND_GENERATION, std::nullopt, std::nullopt, std::nullopt,
            mask);

        auto const reqId = executor.enqueueRequest(std::move(request));
        std::optional<texec::Result> finalResult;
        while (!finalResult)
        {
            auto responses = executor.awaitResponses(std::chrono::milliseconds(100));
            for (auto const& response : responses)
            {
                if (response.getRequestId() == reqId)
                {
                    if (response.hasError())
                    {
                        throw std::runtime_error(response.getErrorMsg());
                    }
                    auto const& result = response.getResult();
                    if (result.isFinal)
                    {
                        finalResult = result;
                        break;
                    }
                }
            }
        }

        int beam = 0;
        for (auto const& tokens : finalResult->outputTokenIds)
        {
            std::cout << "beam[" << beam++ << "]=" << decode(tokens, targetSymbols) << "\n";
        }
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "[indicxlit-cpp] error: " << e.what() << "\n";
        return 1;
    }
}
