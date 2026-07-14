cmake_minimum_required(VERSION 3.20)
project(indicxlit_trtllm_executor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRTLLM_ROOT "/home/ubuntu/TensorRT-LLM" CACHE PATH "TensorRT-LLM source root")
set(TRTLLM_BUILD "${TRTLLM_ROOT}/cpp/build" CACHE PATH "TensorRT-LLM C++ build directory")
set(TENSORRT_ROOT "/tmp/tensorrt_sdk_10_13" CACHE PATH "TensorRT SDK root")

add_executable(indicxlit_executor_smoke src/indicxlit_executor_smoke.cc)

target_include_directories(indicxlit_executor_smoke PRIVATE
  "${TRTLLM_ROOT}/cpp/include"
  "${TRTLLM_ROOT}/cpp/build/_deps/json-src/include"
  "/usr/local/cuda/include"
)

find_library(TRTLLM_LIB tensorrt_llm REQUIRED
  PATHS "${TRTLLM_BUILD}/tensorrt_llm" NO_DEFAULT_PATH)
find_library(TRTLLM_PLUGIN_LIB nvinfer_plugin_tensorrt_llm REQUIRED
  PATHS "${TRTLLM_BUILD}/tensorrt_llm/plugins" NO_DEFAULT_PATH)
find_library(NVINFER_LIB nvinfer REQUIRED
  PATHS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" NO_DEFAULT_PATH)

target_link_libraries(indicxlit_executor_smoke PRIVATE
  "${TRTLLM_LIB}"
  "${TRTLLM_PLUGIN_LIB}"
  "${NVINFER_LIB}"
  cuda
)

set_target_properties(indicxlit_executor_smoke PROPERTIES
  BUILD_RPATH "${TRTLLM_BUILD}/tensorrt_llm;${TRTLLM_BUILD}/tensorrt_llm/plugins;${TRTLLM_BUILD}/tensorrt_llm/kernels/decoderMaskedMultiheadAttention;${TRTLLM_BUILD}/tensorrt_llm/thop;${TRTLLM_BUILD}/tensorrt_llm/runtime/utils"
)
