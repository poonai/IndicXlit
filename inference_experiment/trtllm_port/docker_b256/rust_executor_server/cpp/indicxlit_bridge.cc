#include "indicxlit_bridge.h"

#include "direct_enc_dec_runner.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace
{
constexpr int32_t kBosId = 0;
constexpr int32_t kPadId = 1;
constexpr int32_t kEosId = 2;
constexpr int32_t kUnkId = 3;

char* copyCString(std::string const& value)
{
    auto* out = new char[value.size() + 1];
    std::memcpy(out, value.c_str(), value.size() + 1);
    return out;
}

void setError(char** errorOut, std::string const& message)
{
    if (errorOut != nullptr)
    {
        *errorOut = copyCString(message);
    }
}

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

std::string jsonQuote(std::string const& value)
{
    std::ostringstream out;
    out << '"';
    for (char ch : value)
    {
        switch (ch)
        {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
    return out.str();
}

std::string jsonArray(std::vector<std::string> const& values)
{
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
        {
            out << ',';
        }
        out << jsonQuote(values[i]);
    }
    out << ']';
    return out.str();
}

} // namespace

struct IndicXlitHandle
{
    fs::path assetRoot;
    fs::path langList;
    std::vector<std::string> sourceSymbols;
    std::unordered_map<std::string, int32_t> tokenToId;
    std::unordered_map<std::string, std::vector<std::string>> targetSymbolCache;
    std::unique_ptr<indicxlit::DirectEncDecRunner> runner;

    IndicXlitHandle(fs::path engineDir, fs::path assets, int32_t maxBatchSize, int32_t maxBeamWidth,
        int32_t maxNumTokens, bool useStaticScheduler)
        : assetRoot(std::move(assets))
        , langList(assetRoot / "lang_list.txt")
        , sourceSymbols(makeSymbols(assetRoot / "corpus-bin" / "dict.en.txt", langList))
        , tokenToId(makeTokenToId(sourceSymbols))
    {
        if (!initTrtLlmPlugins())
        {
            throw std::runtime_error("failed to initialize TensorRT-LLM plugins");
        }

        runner = std::make_unique<indicxlit::DirectEncDecRunner>(engineDir, maxBatchSize, maxBeamWidth, maxNumTokens);
        (void) useStaticScheduler;
    }

    std::vector<int32_t> encodeWord(std::string const& word, std::string const& targetLang) const
    {
        std::vector<int32_t> ids;
        auto const langToken = "__" + targetLang + "__";
        auto langIt = tokenToId.find(langToken);
        ids.push_back(langIt != tokenToId.end() ? langIt->second : kUnkId);

        auto lowered = lowerAscii(word);
        for (char ch : lowered)
        {
            std::string piece(1, ch);
            auto pieceIt = tokenToId.find(piece);
            ids.push_back(pieceIt != tokenToId.end() ? pieceIt->second : kUnkId);
        }
        ids.push_back(kEosId);
        return ids;
    }

    std::vector<std::string> const& targetSymbols(std::string const& targetLang)
    {
        auto it = targetSymbolCache.find(targetLang);
        if (it == targetSymbolCache.end())
        {
            it = targetSymbolCache
                     .emplace(targetLang,
                         makeSymbols(assetRoot / "corpus-bin" / ("dict." + targetLang + ".txt"), langList))
                     .first;
        }
        return it->second;
    }

    std::string decode(std::vector<int32_t> const& tokens, std::vector<std::string> const& symbols) const
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
            spaced << ((token >= 0 && static_cast<size_t>(token) < symbols.size()) ? symbols[token] : "<unk>");
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

    std::vector<std::string> inferBatch(
        std::vector<std::string> const& words, std::string const& targetLang, int32_t maxTokens, int32_t beamWidth, int32_t topk)
    {
        std::vector<std::vector<int32_t>> encoded;
        encoded.reserve(words.size());
        for (auto const& word : words)
        {
            encoded.push_back(encodeWord(word, targetLang));
        }

        auto tokenOutputs = runner->infer(encoded, maxTokens, beamWidth);
        auto const& symbols = targetSymbols(targetLang);
        std::vector<std::string> outputs;
        outputs.reserve(tokenOutputs.size());
        for (auto const& item : tokenOutputs)
        {
            std::vector<std::string> beams;
            for (auto const& beam : item)
            {
                if (static_cast<int32_t>(beams.size()) >= topk)
                {
                    break;
                }
                beams.push_back(decode(beam, symbols));
            }
            outputs.push_back(jsonArray(beams));
        }
        return outputs;
    }
};

extern "C"
{
IndicXlitHandle* indicxlit_create(IndicXlitConfig const* config, char** error_out)
{
    try
    {
        if (config == nullptr || config->engine_dir == nullptr || config->asset_root == nullptr)
        {
            throw std::runtime_error("missing IndicXlitConfig fields");
        }
        return new IndicXlitHandle(config->engine_dir, config->asset_root, config->max_batch_size,
            config->max_beam_width, config->max_num_tokens, config->use_static_scheduler != 0);
    }
    catch (std::exception const& e)
    {
        setError(error_out, e.what());
        return nullptr;
    }
}

void indicxlit_destroy(IndicXlitHandle* handle)
{
    delete handle;
}

int32_t indicxlit_infer_batch(IndicXlitHandle* handle, char const* const* words, size_t word_count,
    char const* target_lang, int32_t max_tokens, int32_t beam_width, int32_t topk, char*** outputs_out,
    size_t* output_count_out, char** error_out)
{
    try
    {
        if (handle == nullptr || words == nullptr || target_lang == nullptr || outputs_out == nullptr
            || output_count_out == nullptr)
        {
            throw std::runtime_error("invalid infer_batch arguments");
        }

        std::vector<std::string> inputWords;
        inputWords.reserve(word_count);
        for (size_t i = 0; i < word_count; ++i)
        {
            inputWords.emplace_back(words[i] != nullptr ? words[i] : "");
        }

        auto outputs = handle->inferBatch(inputWords, target_lang, max_tokens, beam_width, topk);
        auto** raw = new char*[outputs.size()];
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            raw[i] = copyCString(outputs[i]);
        }
        *outputs_out = raw;
        *output_count_out = outputs.size();
        return 0;
    }
    catch (std::exception const& e)
    {
        setError(error_out, e.what());
        return 1;
    }
}

void indicxlit_free_string(char* value)
{
    delete[] value;
}

void indicxlit_free_string_array(char** values, size_t count)
{
    if (values == nullptr)
    {
        return;
    }
    for (size_t i = 0; i < count; ++i)
    {
        delete[] values[i];
    }
    delete[] values;
}
}
