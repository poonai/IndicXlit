ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:25.06-trtllm-python-py3
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/IndicXlit

COPY ["model_repository", "/workspace/IndicXlit/inference experiment/trtllm_port/triton_indicxlit/model_repository"]
COPY ["engines/trtllm_engines_en_hi_beam5_triton_fp16_b256", "/workspace/IndicXlit/inference experiment/trtllm_port/artifacts/trtllm_engines_en_hi_beam5_triton_fp16_b256"]
COPY ["model_assets/en2indic", "/workspace/IndicXlit/app/ai4bharat/transliteration/transformer/models/en2indic"]
COPY scripts /opt/indicxlit/scripts
COPY k6 /opt/indicxlit/k6

RUN chmod +x /opt/indicxlit/scripts/run_server.sh /opt/indicxlit/scripts/smoke_core.py

EXPOSE 8000 8002

ENTRYPOINT ["/opt/indicxlit/scripts/run_server.sh"]
