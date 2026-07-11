ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
FROM ${BASE_IMAGE} AS backend-builder

ARG TRITON_TAG=r25.06
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*
COPY cpp_backend /src/indicxlit-native
RUN cmake -S /src/indicxlit-native -B /tmp/native-build \
      -DCMAKE_BUILD_TYPE=Release -DTRITON_TAG=${TRITON_TAG} -DCMAKE_INSTALL_PREFIX=/tmp/native-install \
    && cmake --build /tmp/native-build --parallel "$(nproc)" \
    && cmake --install /tmp/native-build

FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/IndicXlit

ENV INDICXLIT_MODEL_ROOT=/models/assets/en2indic
COPY model_repository /models/indicxlit
COPY engines/trtllm_engines_en_hi_beam5_triton_fp16_b256 /models/engines/b256
COPY model_assets/en2indic /models/assets/en2indic
COPY --from=backend-builder /tmp/native-install/backends/indicxlit_preprocess /opt/tritonserver/backends/indicxlit_preprocess
COPY --from=backend-builder /tmp/native-install/backends/indicxlit_postprocess /opt/tritonserver/backends/indicxlit_postprocess
COPY scripts /opt/indicxlit/scripts
COPY k6 /opt/indicxlit/k6

RUN chmod +x /opt/indicxlit/scripts/run_server.sh /opt/indicxlit/scripts/smoke_core.py

EXPOSE 8000 8001 8002

ENTRYPOINT ["/opt/indicxlit/scripts/run_server.sh"]
