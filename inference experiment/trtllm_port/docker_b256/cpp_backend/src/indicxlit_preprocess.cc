#include "backend.hpp"

#include <memory>
#include <unordered_set>

using namespace indicxlit;

namespace {
struct State {
  std::unordered_map<std::string, int32_t> ids;
  std::unordered_set<std::string> languages;
  State()
  {
    const auto root = ModelRoot();
    const auto lang_path = root + "/lang_list.txt";
    const auto vocab = Vocabulary(root + "/v1.0/corpus-bin/dict.en.txt", lang_path);
    for (size_t i = 0; i < vocab.size(); ++i) ids.emplace(vocab[i], i);
    for (const auto& token : Lines(lang_path)) languages.emplace(token);
  }
};

State* g_state = nullptr;

template <typename T>
TRITONSERVER_Error* Scalar(TRITONBACKEND_Request* request, const char* name, T fallback, T* value)
{
  TRITONBACKEND_Input* input = nullptr;
  auto* error = TRITONBACKEND_RequestInput(request, name, &input);
  if (error != nullptr) { TRITONSERVER_ErrorDelete(error); *value = fallback; return nullptr; }
  const void* raw; size_t bytes; const int64_t* shape; uint32_t dims; TRITONSERVER_DataType type;
  RETURN_IF_ERROR(CpuInput(request, name, &raw, &bytes, &shape, &dims, &type));
  if (bytes < sizeof(T)) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "scalar input is empty");
  std::memcpy(value, raw, sizeof(T));
  return nullptr;
}

TRITONSERVER_Error* Process(TRITONBACKEND_Request* request, TRITONBACKEND_Response* response)
{
  const void* raw; size_t bytes; const int64_t* shape; uint32_t dims; TRITONSERVER_DataType type;
  RETURN_IF_ERROR(CpuInput(request, "TEXT", &raw, &bytes, &shape, &dims, &type));
  auto words = DecodeBytes(raw, bytes);
  if (words.empty()) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "TEXT is empty");

  std::string lang = "hi";
  TRITONBACKEND_Input* lang_input = nullptr;
  auto* lang_error = TRITONBACKEND_RequestInput(request, "TARGET_LANG", &lang_input);
  if (lang_error == nullptr) {
    RETURN_IF_ERROR(CpuInput(request, "TARGET_LANG", &raw, &bytes, &shape, &dims, &type));
    auto values = DecodeBytes(raw, bytes);
    if (!values.empty() && !values[0].empty()) lang = values[0];
  } else { TRITONSERVER_ErrorDelete(lang_error); }
  std::transform(lang.begin(), lang.end(), lang.begin(), [](unsigned char c){ return std::tolower(c); });
  if (!g_state->languages.count(lang))
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, ("unsupported target language: " + lang).c_str());

  int32_t output_len = 32;
  RETURN_IF_ERROR(Scalar(request, "REQUEST_OUTPUT_LEN", int32_t{32}, &output_len));
  output_len = std::clamp(output_len, int32_t{1}, int32_t{63});

  std::vector<std::vector<int32_t>> rows;
  size_t max_len = 0;
  for (auto word : words) {
    std::transform(word.begin(), word.end(), word.begin(), [](unsigned char c){ return std::tolower(c); });
    std::vector<int32_t> row;
    row.push_back(g_state->ids.count("__" + lang + "__") ? g_state->ids.at("__" + lang + "__") : kUnk);
    for (const auto& token : Utf8Chars(word)) row.push_back(g_state->ids.count(token) ? g_state->ids.at(token) : kUnk);
    row.push_back(kEos);
    max_len = std::max(max_len, row.size());
    rows.emplace_back(std::move(row));
  }
  const size_t batch = rows.size();
  std::vector<int32_t> ids(batch * max_len, kPad), lengths(batch), decoder(batch, kEos), ones(batch, 1), output_lengths(batch, output_len), eos(batch, kEos), pad(batch, kPad);
  std::vector<uint8_t> mask(batch * output_len * max_len, 0);
  for (size_t b = 0; b < batch; ++b) {
    lengths[b] = rows[b].size();
    std::copy(rows[b].begin(), rows[b].end(), ids.begin() + b * max_len);
    for (int32_t t = 0; t < output_len; ++t)
      std::fill(mask.begin() + (b * output_len + t) * max_len,
                mask.begin() + (b * output_len + t) * max_len + rows[b].size(), 1);
  }
  const std::vector<int64_t> token_shape{static_cast<int64_t>(batch), static_cast<int64_t>(max_len)};
  const std::vector<int64_t> scalar_shape{static_cast<int64_t>(batch), 1};
  RETURN_IF_ERROR(Output(response, "INPUT_ID", TRITONSERVER_TYPE_INT32, token_shape, ids.data(), ids.size() * 4));
  const std::vector<int64_t> mask_shape{static_cast<int64_t>(batch), output_len, static_cast<int64_t>(max_len)};
  RETURN_IF_ERROR(Output(response, "CROSS_ATTENTION_MASK", TRITONSERVER_TYPE_BOOL, mask_shape, mask.data(), mask.size()));
  RETURN_IF_ERROR(Output(response, "REQUEST_INPUT_LEN", TRITONSERVER_TYPE_INT32, scalar_shape, lengths.data(), lengths.size() * 4));
  RETURN_IF_ERROR(Output(response, "DECODER_INPUT_ID", TRITONSERVER_TYPE_INT32, scalar_shape, decoder.data(), decoder.size() * 4));
  RETURN_IF_ERROR(Output(response, "REQUEST_DECODER_INPUT_LEN", TRITONSERVER_TYPE_INT32, scalar_shape, ones.data(), ones.size() * 4));
  RETURN_IF_ERROR(Output(response, "REQUEST_OUTPUT_LEN", TRITONSERVER_TYPE_INT32, scalar_shape, output_lengths.data(), output_lengths.size() * 4));
  RETURN_IF_ERROR(Output(response, "OUT_END_ID", TRITONSERVER_TYPE_INT32, scalar_shape, eos.data(), eos.size() * 4));
  RETURN_IF_ERROR(Output(response, "OUT_PAD_ID", TRITONSERVER_TYPE_INT32, scalar_shape, pad.data(), pad.size() * 4));
  std::vector<std::string> langs(batch, lang);
  auto encoded_langs = EncodeBytes(langs);
  RETURN_IF_ERROR(Output(response, "TARGET_LANG", TRITONSERVER_TYPE_BYTES, scalar_shape, encoded_langs.data(), encoded_langs.size()));
  return nullptr;
}
}  // namespace

extern "C" TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend*)
{
  try { g_state = new State(); } catch (const std::exception& e) { return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, e.what()); }
  return nullptr;
}
extern "C" TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend*) { delete g_state; g_state = nullptr; return nullptr; }
extern "C" TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests, uint32_t count)
{
  uint64_t batch_start_ns = 0;
  SET_TIMESTAMP(batch_start_ns);
  size_t total_batch_size = 0;
  for (uint32_t i = 0; i < count; ++i) {
    total_batch_size += RequestBatchSize(requests[i]);
    uint64_t exec_start_ns = 0, compute_start_ns = 0, compute_end_ns = 0, exec_end_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    TRITONBACKEND_Response* response = nullptr;
    auto* error = TRITONBACKEND_ResponseNew(&response, requests[i]);
    SET_TIMESTAMP(compute_start_ns);
    if (error == nullptr) error = Process(requests[i], response);
    SET_TIMESTAMP(compute_end_ns);
    const bool success = error == nullptr;
    if (error != nullptr) { SendError(response, TRITONSERVER_ErrorMessage(error)); TRITONSERVER_ErrorDelete(error); }
    else LOG_IF_ERROR(TRITONBACKEND_ResponseSend(response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr), "send response");
    SET_TIMESTAMP(exec_end_ns);
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance, requests[i], success, exec_start_ns, compute_start_ns,
            compute_end_ns, exec_end_ns),
        "report preprocess request statistics");
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL), "release request");
  }
  uint64_t batch_end_ns = 0;
  SET_TIMESTAMP(batch_end_ns);
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance, total_batch_size, batch_start_ns, batch_start_ns,
          batch_end_ns, batch_end_ns),
      "report preprocess batch statistics");
  return nullptr;
}
