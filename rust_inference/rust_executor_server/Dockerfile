# syntax=docker/dockerfile:1.6
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3
FROM ${BASE_IMAGE} AS build

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ninja-build \
 && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable

ENV PATH=/root/.cargo/bin:${PATH}

COPY Cargo.toml build.rs /build/rust/
COPY src /build/rust/src
COPY cpp /build/rust/cpp
COPY trtllm_src /opt/trtllm-src

RUN cmake -S /build/rust/cpp -B /build/rust/cpp_build \
      -G Ninja \
      -DTRTLLM_INCLUDE_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/include \
      -DTRTLLM_ROOT=/opt/trtllm-src \
      -DTRTLLM_LIB_DIR=/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs \
      -DTENSORRT_LIB_DIR=/usr/local/tensorrt/targets/x86_64-linux-gnu/lib \
      -DTENSORRT_INCLUDE_DIR=/usr/local/tensorrt/targets/x86_64-linux-gnu/include \
 && cmake --build /build/rust/cpp_build --target indicxlit_trtllm_bridge \
 && mkdir -p /build/rust/native \
 && cp /build/rust/cpp_build/libindicxlit_trtllm_bridge.so /build/rust/native/libindicxlit_trtllm_bridge.so

WORKDIR /build/rust
RUN cargo build --release

FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1
ENV INDICXLIT_ENGINE_DIR=/models/engines/en_hi_beam5_fp16_b256_continuous_decoder_kv
ENV INDICXLIT_MODEL_ROOT=/models/assets/en2indic
ENV LD_LIBRARY_PATH=/opt/indicxlit/lib:/usr/local/lib/python3.12/dist-packages/tensorrt_llm/libs:/usr/local/tensorrt/targets/x86_64-linux-gnu/lib:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
ENV INDICXLIT_STATIC_SCHEDULER=1
ENV INDICXLIT_MAX_BATCH_SIZE=256
ENV INDICXLIT_BATCH_DELAY_US=2000
ENV INDICXLIT_WORKERS=1
ENV INDICXLIT_RESCORE=1
ENV INDICXLIT_RESCORE_ALPHA=0.9

COPY --from=build /build/rust/target/release/indicxlit-rust-executor-server /opt/indicxlit/bin/indicxlit-rust-executor-server
COPY --from=build /build/rust/native/libindicxlit_trtllm_bridge.so /opt/indicxlit/lib/libindicxlit_trtllm_bridge.so
COPY engine /models/engines/en_hi_beam5_fp16_b256_continuous_decoder_kv
COPY assets /models/assets/en2indic

EXPOSE 8000
ENTRYPOINT ["/opt/indicxlit/bin/indicxlit-rust-executor-server"]
