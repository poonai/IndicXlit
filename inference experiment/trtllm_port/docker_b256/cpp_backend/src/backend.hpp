#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/core/tritonbackend.h"

namespace indicxlit {

constexpr int32_t kBos = 0, kPad = 1, kEos = 2, kUnk = 3;

inline std::vector<std::string> Lines(const std::string& path, bool dictionary = false)
{
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::vector<std::string> out;
  for (std::string line; std::getline(in, line);) {
    if (line.empty()) continue;
    if (dictionary) {
      const auto split = line.rfind(' ');
      if (split != std::string::npos) line.resize(split);
    }
    out.push_back(line);
  }
  return out;
}

inline std::vector<std::string> Vocabulary(const std::string& dict, const std::string& langs)
{
  std::vector<std::string> out{"<s>", "<pad>", "</s>", "<unk>"};
  auto symbols = Lines(dict, true);
  auto language_tokens = Lines(langs);
  out.insert(out.end(), symbols.begin(), symbols.end());
  out.insert(out.end(), language_tokens.begin(), language_tokens.end());
  return out;
}

inline std::string JsonEscape(const std::string& value)
{
  std::ostringstream out;
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 0x20) out << "\\u" << std::hex << static_cast<int>(c);
    else out << c;
  }
  return out.str();
}

inline std::vector<std::string> Utf8Chars(const std::string& value)
{
  std::vector<std::string> chars;
  for (size_t i = 0; i < value.size();) {
    const unsigned char c = value[i];
    size_t width = (c < 0x80) ? 1 : ((c >> 5) == 0x6 ? 2 : ((c >> 4) == 0xe ? 3 : 4));
    width = std::min(width, value.size() - i);
    chars.emplace_back(value.substr(i, width));
    i += width;
  }
  return chars;
}

inline TRITONSERVER_Error* CpuInput(TRITONBACKEND_Request* request, const char* name,
                                    const void** buffer, size_t* bytes,
                                    const int64_t** shape, uint32_t* dims,
                                    TRITONSERVER_DataType* datatype)
{
  TRITONBACKEND_Input* input = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, name, &input));
  uint64_t byte_size = 0;
  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(input, nullptr, datatype, shape, dims, &byte_size, &buffer_count));
  if (buffer_count != 1) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, "fragmented input buffers are unsupported");
  TRITONSERVER_MemoryType memory_type;
  int64_t memory_id;
  RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(input, 0, buffer, bytes, &memory_type, &memory_id));
  if (memory_type != TRITONSERVER_MEMORY_CPU && memory_type != TRITONSERVER_MEMORY_CPU_PINNED)
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, "native text backends require CPU tensors");
  return nullptr;
}

inline std::vector<std::string> DecodeBytes(const void* raw, size_t bytes)
{
  const char* p = static_cast<const char*>(raw);
  const char* end = p + bytes;
  std::vector<std::string> values;
  while (p + 4 <= end) {
    uint32_t n;
    std::memcpy(&n, p, 4);
    p += 4;
    if (p + n > end) throw std::runtime_error("invalid Triton BYTES tensor");
    values.emplace_back(p, n);
    p += n;
  }
  if (p != end) throw std::runtime_error("invalid Triton BYTES framing");
  return values;
}

inline std::vector<char> EncodeBytes(const std::vector<std::string>& values)
{
  size_t size = 0;
  for (const auto& value : values) size += 4 + value.size();
  std::vector<char> out(size);
  char* p = out.data();
  for (const auto& value : values) {
    const uint32_t n = value.size();
    std::memcpy(p, &n, 4); p += 4;
    std::memcpy(p, value.data(), n); p += n;
  }
  return out;
}

inline TRITONSERVER_Error* Output(TRITONBACKEND_Response* response, const char* name,
                                  TRITONSERVER_DataType type, const std::vector<int64_t>& shape,
                                  const void* data, size_t bytes)
{
  TRITONBACKEND_Output* output;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, name, type, shape.data(), shape.size()));
  void* dst;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &dst, bytes, &memory_type, &memory_id));
  if (bytes) std::memcpy(dst, data, bytes);
  return nullptr;
}

inline void SendError(TRITONBACKEND_Response*& response, const std::string& message)
{
  auto* error = TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, message.c_str());
  LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error), "send error response");
  response = nullptr;
}

inline std::string ModelRoot()
{
  const char* value = std::getenv("INDICXLIT_MODEL_ROOT");
  return value ? value : "/models/assets/en2indic";
}

inline size_t RequestBatchSize(TRITONBACKEND_Request* request)
{
  TRITONBACKEND_Input* input = nullptr;
  auto* error = TRITONBACKEND_RequestInputByIndex(request, 0, &input);
  if (error != nullptr) { TRITONSERVER_ErrorDelete(error); return 1; }
  if (input == nullptr) return 1;
  const int64_t* shape = nullptr;
  uint32_t dims = 0;
  error = TRITONBACKEND_InputProperties(input, nullptr, nullptr, &shape, &dims, nullptr, nullptr);
  if (error != nullptr) { TRITONSERVER_ErrorDelete(error); return 1; }
  if (shape == nullptr || dims == 0 || shape[0] < 1) return 1;
  return static_cast<size_t>(shape[0]);
}

}  // namespace indicxlit
