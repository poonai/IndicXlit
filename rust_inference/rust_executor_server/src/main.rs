mod ffi;

use anyhow::{anyhow, Context, Result};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use crossbeam_channel::{unbounded, Receiver, RecvTimeoutError, Sender};
use metrics::{counter, gauge, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::env;
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tracing::{error, info};

#[derive(Clone)]
struct AppState {
    tx: Sender<WorkItem>,
    prometheus: PrometheusHandle,
    default_lang: String,
    default_max_tokens: i32,
    default_beam_width: i32,
    default_topk: i32,
}

struct WorkItem {
    words: Vec<String>,
    target_lang: String,
    max_tokens: i32,
    beam_width: i32,
    topk: i32,
    enqueued_at: Instant,
    response: oneshot::Sender<WorkResult>,
}

struct WorkResult {
    worker_id: usize,
    result: Result<Vec<String>, String>,
}

#[derive(Debug, Deserialize)]
struct TritonRequest {
    #[serde(default)]
    inputs: Vec<TritonInput>,
}

#[derive(Debug, Deserialize)]
struct TritonInput {
    name: String,
    #[serde(default)]
    data: Value,
}

#[derive(Debug, Serialize)]
struct TritonOutput {
    name: &'static str,
    datatype: &'static str,
    shape: [usize; 2],
    data: Vec<String>,
}

fn input_string(req: &TritonRequest, name: &str) -> Option<String> {
    req.inputs.iter().find(|input| input.name == name).and_then(|input| match &input.data {
        Value::Array(items) => items.first().and_then(Value::as_str).map(str::to_string),
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    })
}

fn input_strings(req: &TritonRequest, name: &str) -> Vec<String> {
    req.inputs
        .iter()
        .find(|input| input.name == name)
        .map(|input| match &input.data {
            Value::Array(items) => items
                .iter()
                .filter_map(|item| match item {
                    Value::String(value) => Some(value.clone()),
                    Value::Number(value) => Some(value.to_string()),
                    _ => None,
                })
                .collect(),
            Value::String(value) => vec![value.clone()],
            Value::Number(value) => vec![value.to_string()],
            _ => Vec::new(),
        })
        .unwrap_or_default()
}

fn input_i32(req: &TritonRequest, name: &str) -> Option<i32> {
    input_string(req, name).and_then(|value| value.parse::<i32>().ok())
}

async fn health() -> Json<Value> {
    Json(json!({"health": "ready"}))
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        state.prometheus.render(),
    )
        .into_response()
}

async fn infer(
    State(state): State<AppState>,
    Json(req): Json<TritonRequest>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let request_started = Instant::now();
    let texts = input_strings(&req, "text_input");
    let mut words: Vec<String> = texts
        .iter()
        .flat_map(|text| text.lines())
        .map(str::trim)
        .filter(|word| !word.is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    if words.is_empty() {
        words = vec![String::new()];
    }
    let word_count = words.len();
    let target_lang = input_string(&req, "target_lang").unwrap_or_else(|| state.default_lang.clone());
    let max_tokens = input_i32(&req, "max_tokens").unwrap_or(state.default_max_tokens);
    let beam_width = input_i32(&req, "beam_width").unwrap_or(state.default_beam_width);
    let topk = input_i32(&req, "topk").unwrap_or(state.default_topk);

    let (response_tx, response_rx) = oneshot::channel();
    state
        .tx
        .send(WorkItem {
            words,
            target_lang,
            max_tokens,
            beam_width,
            topk,
            enqueued_at: Instant::now(),
            response: response_tx,
        })
        .map_err(|_| {
            record_http_metrics("error", "none", word_count, request_started.elapsed());
            (StatusCode::SERVICE_UNAVAILABLE, "batch worker stopped".to_string())
        })?;

    let work_result = response_rx.await.map_err(|_| {
        record_http_metrics("error", "none", word_count, request_started.elapsed());
        (StatusCode::SERVICE_UNAVAILABLE, "batch worker dropped response".to_string())
    })?;
    let worker_label = format!("worker-{}", work_result.worker_id);
    let candidates = work_result.result.map_err(|error| {
        record_http_metrics("error", &worker_label, word_count, request_started.elapsed());
        (StatusCode::INTERNAL_SERVER_ERROR, error)
    })?;

    let best: Vec<String> = candidates
        .iter()
        .map(|candidate_json| {
            serde_json::from_str::<Vec<String>>(candidate_json)
                .ok()
                .and_then(|values| values.into_iter().next())
                .unwrap_or_default()
        })
        .collect();
    let output_count = candidates.len();
    record_http_metrics("ok", &worker_label, output_count, request_started.elapsed());

    Ok(Json(json!({
        "model_name": "indicxlit",
        "outputs": [
            TritonOutput {
                name: "text_output",
                datatype: "BYTES",
                shape: [output_count, 1],
                data: best,
            },
            TritonOutput {
                name: "candidates_json",
                datatype: "BYTES",
                shape: [output_count, 1],
                data: candidates,
            }
        ]
    })))
}

fn pin_worker_to_cpu(worker_id: usize) {
    let Some(core_ids) = core_affinity::get_core_ids() else {
        info!(worker_id, "cpu affinity unavailable");
        return;
    };
    if core_ids.is_empty() {
        info!(worker_id, "no cpu cores reported for affinity");
        return;
    }
    let core_id = core_ids[worker_id % core_ids.len()];
    if core_affinity::set_for_current(core_id) {
        let worker_label = format!("worker-{worker_id}");
        gauge!("indicxlit_worker_cpu_core", "worker" => worker_label).set(core_id.id as f64);
        info!(worker_id, cpu_core = core_id.id, "pinned worker to cpu core");
    } else {
        info!(worker_id, cpu_core = core_id.id, "failed to pin worker to cpu core");
    }
}

fn record_http_metrics(status: &'static str, worker: &str, words: usize, elapsed: Duration) {
    counter!("indicxlit_http_requests_total", "status" => status, "worker" => worker.to_string()).increment(1);
    counter!("indicxlit_http_request_words_total", "status" => status, "worker" => worker.to_string()).increment(words as u64);
    histogram!("indicxlit_http_request_duration_seconds", "status" => status, "worker" => worker.to_string())
        .record(elapsed.as_secs_f64());
    histogram!("indicxlit_http_request_words", "status" => status, "worker" => worker.to_string()).record(words as f64);
}

fn record_batch_metrics(
    status: &'static str,
    mode: &'static str,
    worker: &str,
    request_count: usize,
    word_count: usize,
    elapsed: Duration,
) {
    counter!("indicxlit_batches_total", "status" => status, "mode" => mode, "worker" => worker.to_string()).increment(1);
    counter!("indicxlit_batch_words_total", "status" => status, "mode" => mode, "worker" => worker.to_string()).increment(word_count as u64);
    histogram!("indicxlit_batch_inference_duration_seconds", "status" => status, "mode" => mode, "worker" => worker.to_string())
        .record(elapsed.as_secs_f64());
    histogram!("indicxlit_batch_requests", "status" => status, "mode" => mode, "worker" => worker.to_string()).record(request_count as f64);
    histogram!("indicxlit_batch_words", "status" => status, "mode" => mode, "worker" => worker.to_string()).record(word_count as f64);
}

fn run_batcher(
    worker_id: usize,
    mut engine: ffi::Engine,
    rx: Receiver<WorkItem>,
    max_batch_size: usize,
    batch_delay: Duration,
) {
    pin_worker_to_cpu(worker_id);
    let worker_label = format!("worker-{worker_id}");
    let mut pending: Option<WorkItem> = None;
    loop {
        let first = match pending.take() {
            Some(item) => item,
            None => match rx.recv() {
                Ok(item) => item,
                Err(_) => break,
            },
        };
        let started = Instant::now();
        let mut batch_word_count = first.words.len();
        let mut batch = vec![first];
        while batch_word_count < max_batch_size && started.elapsed() < batch_delay {
            let remaining = batch_delay.saturating_sub(started.elapsed());
            match rx.recv_timeout(remaining) {
                Ok(item) => {
                    let item_words = item.words.len();
                    if !batch.is_empty() && batch_word_count + item_words > max_batch_size {
                        pending = Some(item);
                        break;
                    }
                    batch_word_count += item_words;
                    batch.push(item);
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }

        let mut flat_words = Vec::new();
        let mut ranges = Vec::with_capacity(batch.len());
        for item in &batch {
            histogram!("indicxlit_queue_wait_duration_seconds", "worker" => worker_label.clone())
                .record(item.enqueued_at.elapsed().as_secs_f64());
            let start = flat_words.len();
            flat_words.extend(item.words.iter().cloned());
            ranges.push(start..flat_words.len());
        }

        let first_cfg = &batch[0];
        let compatible = batch.iter().all(|item| {
            item.target_lang == first_cfg.target_lang
                && item.max_tokens == first_cfg.max_tokens
                && item.beam_width == first_cfg.beam_width
                && item.topk == first_cfg.topk
        });

        if !compatible {
            for item in batch {
                let word_count = item.words.len();
                let infer_started = Instant::now();
                let result = engine.infer_batch(
                    &item.words,
                    &item.target_lang,
                    item.max_tokens,
                    item.beam_width,
                    item.topk,
                );
                record_batch_metrics(
                    if result.is_ok() { "ok" } else { "error" },
                    "single_incompatible",
                    &worker_label,
                    1,
                    word_count,
                    infer_started.elapsed(),
                );
                let _ = item.response.send(WorkResult {
                    worker_id,
                    result: result.map_err(|error| error.to_string()),
                });
            }
            continue;
        }

        let infer_started = Instant::now();
        let result = engine.infer_batch(
            &flat_words,
            &first_cfg.target_lang,
            first_cfg.max_tokens,
            first_cfg.beam_width,
            first_cfg.topk,
        );
        record_batch_metrics(
            if result.is_ok() { "ok" } else { "error" },
            "merged",
            &worker_label,
            batch.len(),
            flat_words.len(),
            infer_started.elapsed(),
        );
        match result {
            Ok(outputs) => {
                for (item, range) in batch.into_iter().zip(ranges) {
                    let _ = item.response.send(WorkResult {
                        worker_id,
                        result: Ok(outputs[range].to_vec()),
                    });
                }
            }
            Err(error) => {
                let message = error.to_string();
                error!(%message, "batch inference failed");
                for item in batch {
                    let _ = item.response.send(WorkResult {
                        worker_id,
                        result: Err(message.clone()),
                    });
                }
            }
        }
    }
}

fn env_string(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_string_any(names: &[&str], default: &str) -> String {
    names
        .iter()
        .find_map(|name| env::var(name).ok())
        .unwrap_or_else(|| default.to_string())
}

fn env_i32(name: &str, default: i32) -> i32 {
    env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()))
        .init();

    let prometheus = PrometheusBuilder::new()
        .install_recorder()
        .context("failed to install Prometheus metrics recorder")?;

    let engine_dir = env_string_any(
        &["INDICXLIT_ENGINE_DIR", "ENGINE_DIR"],
        "/models/engines/en_hi_beam5_fp16_b256_paged_kv",
    );
    let asset_root = env_string("INDICXLIT_MODEL_ROOT", "/models/assets/en2indic");
    let host = env_string("INDICXLIT_HOST", "0.0.0.0");
    let port = env_i32("INDICXLIT_PORT", 8000);
    let max_batch_size = env_i32("INDICXLIT_MAX_BATCH_SIZE", 256);
    let max_beam_width = env_i32("INDICXLIT_MAX_BEAM_WIDTH", 5);
    let max_num_tokens = env_i32("INDICXLIT_MAX_NUM_TOKENS", 8192);
    let batch_delay_us = env_i32("INDICXLIT_BATCH_DELAY_US", 2000);
    let use_static_scheduler = env_i32("INDICXLIT_STATIC_SCHEDULER", 1) != 0;
    let worker_count = env_i32("INDICXLIT_WORKERS", 2).max(1) as usize;

    gauge!("indicxlit_config_max_batch_size").set(max_batch_size as f64);
    gauge!("indicxlit_config_max_beam_width").set(max_beam_width as f64);
    gauge!("indicxlit_config_max_num_tokens").set(max_num_tokens as f64);
    gauge!("indicxlit_config_batch_delay_microseconds").set(batch_delay_us as f64);
    gauge!("indicxlit_config_static_scheduler").set(if use_static_scheduler { 1.0 } else { 0.0 });
    gauge!("indicxlit_config_workers").set(worker_count as f64);

    let (tx, rx) = unbounded();
    for worker_id in 0..worker_count {
        info!(worker_id, "initializing TensorRT-LLM worker");
        let engine = ffi::Engine::new(
            &engine_dir,
            &asset_root,
            max_batch_size,
            max_beam_width,
            max_num_tokens,
            use_static_scheduler,
        )
        .with_context(|| format!("failed to initialize C++ TensorRT-LLM engine for worker {worker_id}"))?;

        let worker_rx = rx.clone();
        std::thread::Builder::new()
            .name(format!("indicxlit-batcher-{worker_id}"))
            .spawn(move || {
                run_batcher(
                    worker_id,
                    engine,
                    worker_rx,
                    max_batch_size as usize,
                    Duration::from_micros(batch_delay_us.max(0) as u64),
                )
            })
            .with_context(|| format!("failed to spawn batcher thread for worker {worker_id}"))?;
    }

    let state = AppState {
        tx,
        prometheus,
        default_lang: env_string("INDICXLIT_LANG", "hi"),
        default_max_tokens: env_i32("INDICXLIT_MAX_TOKENS", 32),
        default_beam_width: env_i32("INDICXLIT_BEAM_WIDTH", 5),
        default_topk: env_i32("INDICXLIT_TOPK", 5),
    };

    let app = Router::new()
        .route("/v2/health/ready", get(health))
        .route("/metrics", get(metrics_endpoint))
        .route("/v2/models/indicxlit/infer", post(infer))
        .with_state(state);

    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|error| anyhow!("invalid bind address: {error}"))?;
    info!(%addr, "starting IndicXlit Rust executor server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
