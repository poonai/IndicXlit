# IndicXlit Rust Inference

Run all commands from the `rust_inference` directory.

## Step 1: Download the checkpoint and build the engine

```bash
sudo ./scripts/download_checkpoint_and_build_engine.sh
```

## Step 2: Build the Rust inference image

```bash
sudo ./scripts/build_rust_executor_image.sh
```

## Step 3: Start the services

```bash
sudo docker compose up -d
```

## Step 4: Run the Dakshina evaluation

```bash
sudo ./scripts/evaluate_dakshina.sh
```

## Step 5: Endpoints

- Index page: `http://localhost:8000/`
- Inference API: `POST http://localhost:8000/v2/models/indicxlit/infer`
- Grafana: `http://localhost:3000/`
- Grafana login: `admin` / `admin`
