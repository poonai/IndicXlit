#pragma once

#include <cstddef>
#include <cstdint>

extern "C"
{
struct IndicXlitHandle;

struct IndicXlitConfig
{
    char const* engine_dir;
    char const* asset_root;
    int32_t max_batch_size;
    int32_t max_beam_width;
    int32_t max_num_tokens;
    int32_t use_static_scheduler;
};

IndicXlitHandle* indicxlit_create(IndicXlitConfig const* config, char** error_out);
void indicxlit_destroy(IndicXlitHandle* handle);

int32_t indicxlit_infer_batch(IndicXlitHandle* handle, char const* const* words, size_t word_count,
    char const* target_lang, int32_t max_tokens, int32_t beam_width, int32_t topk, char*** outputs_out,
    size_t* output_count_out, char** error_out);

void indicxlit_free_string(char* value);
void indicxlit_free_string_array(char** values, size_t count);
}
