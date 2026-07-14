ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:26.02-trtllm-python-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1

WORKDIR /workspace/IndicXlit

ENV INDICXLIT_MODEL_ROOT=/models/assets/en2indic
COPY model_repository /models/indicxlit
COPY engines/trtllm_engines_en_hi_beam5_triton_fp16_b256 /models/engines/b256
COPY model_assets/en2indic /models/assets/en2indic
COPY scripts /opt/indicxlit/scripts
COPY k6 /opt/indicxlit/k6

RUN chmod +x /opt/indicxlit/scripts/run_server.sh /opt/indicxlit/scripts/smoke_core.py /opt/indicxlit/scripts/smoke_ensemble.py

EXPOSE 8000 8001 8002

ENTRYPOINT ["/opt/indicxlit/scripts/run_server.sh"]
