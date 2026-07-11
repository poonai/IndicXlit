#include "backend.hpp"

#include <memory>

using namespace indicxlit;

namespace {
struct State {
  std::unordered_map<std::string, std::vector<std::string>> vocab;
  std::vector<std::string> languages;
  State()
  {
    const auto root = ModelRoot();
    const auto langs = root + "/lang_list.txt";
    for (const auto& lang : Lines(langs)) {
      if (lang == "en") continue;
      vocab.emplace(lang, Vocabulary(root + "/v1.0/corpus-bin/dict." + lang + ".txt", langs));
    }
  }
};
State* g_state = nullptr;

TRITONSERVER_Error* Process(TRITONBACKEND_Request* request, TRITONBACKEND_Response* response)
{
  const void* raw; size_t bytes; const int64_t* shape; uint32_t dims; TRITONSERVER_DataType type;
  RETURN_IF_ERROR(CpuInput(request, "TOKENS_BATCH", &raw, &bytes, &shape, &dims, &type));
  if (dims < 3) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "TOKENS_BATCH must have shape [batch, beams, tokens]");
  const int64_t batch = shape[0], beams = shape[1], tokens = shape[2];
  const auto* ids = static_cast<const int32_t*>(raw);

  const void* length_raw; size_t length_bytes; const int64_t* length_shape;
  uint32_t length_dims; TRITONSERVER_DataType length_type;
  RETURN_IF_ERROR(CpuInput(request, "SEQUENCE_LENGTH", &length_raw, &length_bytes,
                           &length_shape, &length_dims, &length_type));
  if (length_bytes < static_cast<size_t>(batch * beams) * sizeof(int32_t))
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "SEQUENCE_LENGTH batch mismatch");
  const auto* sequence_lengths = static_cast<const int32_t*>(length_raw);

  RETURN_IF_ERROR(CpuInput(request, "TARGET_LANG", &raw, &bytes, &shape, &dims, &type));
  auto langs = DecodeBytes(raw, bytes);
  if (langs.size() == 1 && batch > 1) langs.resize(batch, langs[0]);
  if (langs.size() != static_cast<size_t>(batch)) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "TARGET_LANG batch mismatch");

  int32_t topk = beams;
  TRITONBACKEND_Input* optional = nullptr;
  auto* optional_error = TRITONBACKEND_RequestInput(request, "TOPK", &optional);
  if (optional_error == nullptr) {
    RETURN_IF_ERROR(CpuInput(request, "TOPK", &raw, &bytes, &shape, &dims, &type));
    if (bytes >= 4) std::memcpy(&topk, raw, 4);
  } else { TRITONSERVER_ErrorDelete(optional_error); }
  topk = std::clamp(topk, int32_t{1}, static_cast<int32_t>(beams));

  std::vector<std::string> best, json;
  for (int64_t b = 0; b < batch; ++b) {
    const auto found = g_state->vocab.find(langs[b].empty() ? "hi" : langs[b]);
    if (found == g_state->vocab.end()) return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, "unsupported TARGET_LANG");
    std::vector<std::string> candidates;
    for (int64_t beam = 0; beam < topk; ++beam) {
      std::string text;
      const int64_t sequence_length = std::clamp<int64_t>(
          sequence_lengths[b * beams + beam], 0, tokens);
      for (int64_t t = 0; t < sequence_length; ++t) {
        const int32_t id = ids[(b * beams + beam) * tokens + t];
        if (id == kEos) break;
        if (id == kBos || id == kPad) continue;
        text += (id >= 0 && static_cast<size_t>(id) < found->second.size()) ? found->second[id] : "<unk>";
      }
      candidates.push_back(text);
    }
    best.push_back(candidates.empty() ? "" : candidates[0]);
    std::ostringstream encoded; encoded << '[';
    for (size_t i = 0; i < candidates.size(); ++i) { if (i) encoded << ','; encoded << '\"' << JsonEscape(candidates[i]) << '\"'; }
    encoded << ']'; json.push_back(encoded.str());
  }
  auto best_bytes = EncodeBytes(best), json_bytes = EncodeBytes(json);
  const std::vector<int64_t> output_shape{batch, 1};
  RETURN_IF_ERROR(Output(response, "TEXT_OUTPUT", TRITONSERVER_TYPE_BYTES, output_shape, best_bytes.data(), best_bytes.size()));
  RETURN_IF_ERROR(Output(response, "CANDIDATES_JSON", TRITONSERVER_TYPE_BYTES, output_shape, json_bytes.data(), json_bytes.size()));
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
        "report postprocess request statistics");
    LOG_IF_ERROR(TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL), "release request");
  }
  uint64_t batch_end_ns = 0;
  SET_TIMESTAMP(batch_end_ns);
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance, total_batch_size, batch_start_ns, batch_start_ns,
          batch_end_ns, batch_end_ns),
      "report postprocess batch statistics");
  return nullptr;
}
