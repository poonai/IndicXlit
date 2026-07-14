use anyhow::{anyhow, Context, Result};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;

#[repr(C)]
struct IndicXlitConfig {
    engine_dir: *const c_char,
    asset_root: *const c_char,
    max_batch_size: i32,
    max_beam_width: i32,
    max_num_tokens: i32,
    use_static_scheduler: i32,
}

#[repr(C)]
struct IndicXlitHandle {
    _private: [u8; 0],
}

#[link(name = "indicxlit_trtllm_bridge")]
extern "C" {
    fn indicxlit_create(config: *const IndicXlitConfig, error_out: *mut *mut c_char) -> *mut IndicXlitHandle;
    fn indicxlit_destroy(handle: *mut IndicXlitHandle);
    fn indicxlit_infer_batch(
        handle: *mut IndicXlitHandle,
        words: *const *const c_char,
        word_count: usize,
        target_lang: *const c_char,
        max_tokens: i32,
        beam_width: i32,
        topk: i32,
        outputs_out: *mut *mut *mut c_char,
        output_count_out: *mut usize,
        error_out: *mut *mut c_char,
    ) -> c_int;
    fn indicxlit_free_string(value: *mut c_char);
    fn indicxlit_free_string_array(values: *mut *mut c_char, count: usize);
}

pub struct Engine {
    handle: *mut IndicXlitHandle,
}

unsafe impl Send for Engine {}

impl Engine {
    pub fn new(
        engine_dir: &str,
        asset_root: &str,
        max_batch_size: i32,
        max_beam_width: i32,
        max_num_tokens: i32,
        use_static_scheduler: bool,
    ) -> Result<Self> {
        let engine_dir = CString::new(engine_dir).context("ENGINE_DIR contains an interior NUL byte")?;
        let asset_root = CString::new(asset_root).context("INDICXLIT_MODEL_ROOT contains an interior NUL byte")?;
        let config = IndicXlitConfig {
            engine_dir: engine_dir.as_ptr(),
            asset_root: asset_root.as_ptr(),
            max_batch_size,
            max_beam_width,
            max_num_tokens,
            use_static_scheduler: i32::from(use_static_scheduler),
        };

        let mut error = ptr::null_mut();
        let handle = unsafe { indicxlit_create(&config, &mut error) };
        if handle.is_null() {
            return Err(take_error(error).unwrap_or_else(|| anyhow!("failed to create IndicXlit TensorRT engine")));
        }
        Ok(Self { handle })
    }

    pub fn infer_batch(
        &mut self,
        words: &[String],
        target_lang: &str,
        max_tokens: i32,
        beam_width: i32,
        topk: i32,
    ) -> Result<Vec<String>> {
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let c_words = words
            .iter()
            .map(|word| CString::new(word.as_str()).with_context(|| format!("input word contains NUL byte: {word:?}")))
            .collect::<Result<Vec<_>>>()?;
        let word_ptrs = c_words.iter().map(|word| word.as_ptr()).collect::<Vec<_>>();
        let target_lang = CString::new(target_lang).context("target_lang contains an interior NUL byte")?;

        let mut outputs = ptr::null_mut();
        let mut output_count = 0usize;
        let mut error = ptr::null_mut();
        let code = unsafe {
            indicxlit_infer_batch(
                self.handle,
                word_ptrs.as_ptr(),
                word_ptrs.len(),
                target_lang.as_ptr(),
                max_tokens,
                beam_width,
                topk,
                &mut outputs,
                &mut output_count,
                &mut error,
            )
        };
        if code != 0 {
            return Err(take_error(error).unwrap_or_else(|| anyhow!("IndicXlit TensorRT inference failed with code {code}")));
        }
        if outputs.is_null() || output_count != words.len() {
            if !outputs.is_null() {
                unsafe { indicxlit_free_string_array(outputs, output_count) };
            }
            return Err(anyhow!(
                "IndicXlit TensorRT returned {} outputs for {} inputs",
                output_count,
                words.len()
            ));
        }

        let raw_outputs = unsafe { std::slice::from_raw_parts(outputs, output_count) };
        let result = raw_outputs
            .iter()
            .map(|value| {
                if value.is_null() {
                    Ok(String::new())
                } else {
                    unsafe { CStr::from_ptr(*value) }
                        .to_str()
                        .context("IndicXlit TensorRT returned non-UTF8 output")
                        .map(str::to_string)
                }
            })
            .collect::<Result<Vec<_>>>();
        unsafe { indicxlit_free_string_array(outputs, output_count) };
        result
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { indicxlit_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

fn take_error(error: *mut c_char) -> Option<anyhow::Error> {
    if error.is_null() {
        return None;
    }
    let message = unsafe { CStr::from_ptr(error) }.to_string_lossy().into_owned();
    unsafe { indicxlit_free_string(error) };
    Some(anyhow!(message))
}
