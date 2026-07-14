cmake_minimum_required(VERSION 3.20)
project(indicxlit_trtllm_bridge LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(TRTLLM_ROOT "/opt/trtllm-src" CACHE PATH "Patched TensorRT-LLM source root")
set(TRTLLM_LIB_DIR "/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs" CACHE PATH "TensorRT-LLM library directory")
set(TENSORRT_LIB_DIR "/usr/local/tensorrt/targets/x86_64-linux-gnu/lib" CACHE PATH "TensorRT library directory")
set(TENSORRT_INCLUDE_DIR "/usr/local/tensorrt/targets/x86_64-linux-gnu/include" CACHE PATH "TensorRT include directory")

add_library(indicxlit_trtllm_bridge SHARED
  indicxlit_bridge.cc
  direct_cache_tile.cu
  direct_dynamic_decode.cc
  direct_enc_dec_runner.cc
)
target_include_directories(indicxlit_trtllm_bridge PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${TRTLLM_ROOT}/cpp/include"
  "${TRTLLM_ROOT}/cpp"
  "/usr/local/cuda/include"
  "${TENSORRT_INCLUDE_DIR}"
)

find_library(TRTLLM_LIB tensorrt_llm REQUIRED PATHS "${TRTLLM_LIB_DIR}" NO_DEFAULT_PATH)
find_library(TRTLLM_PLUGIN_LIB nvinfer_plugin_tensorrt_llm REQUIRED PATHS "${TRTLLM_LIB_DIR}" NO_DEFAULT_PATH)
find_library(NVINFER_LIB nvinfer REQUIRED PATHS "${TENSORRT_LIB_DIR}" NO_DEFAULT_PATH)
find_library(CUDART_LIB cudart REQUIRED
  PATHS
    "/usr/local/cuda/lib64"
    "/usr/local/cuda/targets/x86_64-linux/lib"
    "/usr/local/cuda-13.0/targets/x86_64-linux/lib"
  NO_DEFAULT_PATH
)

target_link_libraries(indicxlit_trtllm_bridge PRIVATE
  "${TRTLLM_LIB}"
  "${TRTLLM_PLUGIN_LIB}"
  "${NVINFER_LIB}"
  "${CUDART_LIB}"
  cuda
  nvidia-ml
)

set_target_properties(indicxlit_trtllm_bridge PROPERTIES
  BUILD_RPATH "${TRTLLM_LIB_DIR};${TENSORRT_LIB_DIR}"
  INSTALL_RPATH "${TRTLLM_LIB_DIR};${TENSORRT_LIB_DIR}"
)

add_executable(indicxlit_direct_smoke direct_smoke.cc)
target_link_libraries(indicxlit_direct_smoke PRIVATE
  indicxlit_trtllm_bridge
  "${TRTLLM_LIB}"
  "${TRTLLM_PLUGIN_LIB}"
  "${NVINFER_LIB}"
  "${CUDART_LIB}"
  cuda
  nvidia-ml
)
target_link_options(indicxlit_direct_smoke PRIVATE
  "-Wl,-rpath-link,${TRTLLM_LIB_DIR}"
  "-Wl,-rpath-link,${TENSORRT_LIB_DIR}"
  "-Wl,-rpath-link,/usr/local/cuda/lib64"
  "-Wl,-rpath-link,/usr/local/cuda/targets/x86_64-linux/lib"
  "-Wl,-rpath-link,/usr/local/cuda-13.0/targets/x86_64-linux/lib"
  "-Wl,-rpath-link,/usr/local/cuda/targets/x86_64-linux/lib/stubs"
  "-Wl,-rpath-link,/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs"
  "-Wl,-rpath-link,/usr/lib/x86_64-linux-gnu"
)
set_target_properties(indicxlit_direct_smoke PROPERTIES
  BUILD_RPATH "${TRTLLM_LIB_DIR};${TENSORRT_LIB_DIR};$ORIGIN"
  INSTALL_RPATH "${TRTLLM_LIB_DIR};${TENSORRT_LIB_DIR};$ORIGIN"
)
