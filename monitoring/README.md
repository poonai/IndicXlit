# IndicXlit Monitoring

The server exposes Prometheus metrics at `http://127.0.0.1:8888/metrics`.

Run the app first:

```bash
source .venv/bin/activate
env -u CUDA_HOME -u CUDA_VERSION python app/start_server.py
```

Then start Prometheus and Grafana on a machine with Docker Compose:

```bash
docker compose -f docker-compose.monitoring.yml up
```

Open Grafana at `http://127.0.0.1:3000` with `admin` / `admin`. The `IndicXlit Telemetry` dashboard is provisioned automatically. Prometheus is available at `http://127.0.0.1:9090`.

GPU panels are populated only when `nvidia-smi` is available to the server process. Request GPU time cannot be measured precisely from Flask alone; the dashboard shows request latency, process CPU time per request, GPU utilization, GPU memory, temperature, and power at scrape time.
