"""
Expose Transliteration Engine as an HTTP API.

USAGE:
```
from ai4bharat.transliteration import xlit_server
app, engine = xlit_server.get_app()
app.run(host='0.0.0.0', port=8000)
```
Sample URLs:
    http://localhost:8000/tl/ta/amma
    http://localhost:8000/languages

FORMAT:
    Based on the Varnam API standard
    https://api.varnamproject.com/tl/hi/bharat
"""

from flask import Flask, Response, jsonify, request, make_response
from flask_limiter import Limiter
from uuid import uuid4
from datetime import datetime
import traceback
import enum
import os
import subprocess
import time

import psutil
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from .utils import LANG_CODE_TO_DISPLAY_NAME, RTL_LANG_CODES, LANG_CODE_TO_SCRIPT_CODE, GOOGLE_FONTS, FALLBACK_FONTS

class XlitError(enum.Enum):
    lang_err = "Unsupported langauge ID requested ;( Please check available languages."
    string_err = "String passed is incompatable ;("
    internal_err = "Internal crash ;("
    unknown_err = "Unknown Failure"
    loading_err = "Loading failed ;( Check if metadata/paths are correctly configured."

def get_remote_address() -> str:
    return request.headers.get('X-Forwarded-For', request.remote_addr) or "127.0.0.1"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
)

PROCESS = psutil.Process(os.getpid())
REQUEST_START = "xlit_request_start_time"
REQUEST_CPU_START = "xlit_request_cpu_start"

REQUEST_COUNT = Counter(
    "xlit_http_requests_total",
    "Total HTTP requests handled by the transliteration API.",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "xlit_http_request_duration_seconds",
    "Wall-clock request latency in seconds.",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
REQUEST_CPU_TIME = Histogram(
    "xlit_http_request_cpu_seconds",
    "Process CPU time consumed while handling a request.",
    ["method", "endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
INFERENCE_LATENCY = Histogram(
    "xlit_inference_duration_seconds",
    "Model inference wall-clock latency in seconds.",
    ["direction", "lang_code", "mode"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
)
IN_PROGRESS = Gauge(
    "xlit_http_requests_in_progress",
    "Number of HTTP requests currently in progress.",
)
PROCESS_CPU_PERCENT = Gauge(
    "xlit_process_cpu_percent",
    "Current transliteration server process CPU utilization percentage.",
)
PROCESS_RSS_BYTES = Gauge(
    "xlit_process_resident_memory_bytes",
    "Resident memory used by the transliteration server process.",
)
PROCESS_VMS_BYTES = Gauge(
    "xlit_process_virtual_memory_bytes",
    "Virtual memory used by the transliteration server process.",
)
PROCESS_THREADS = Gauge(
    "xlit_process_threads",
    "Number of process threads used by the transliteration server.",
)
PROCESS_OPEN_FDS = Gauge(
    "xlit_process_open_fds",
    "Number of open file descriptors used by the transliteration server.",
)
GPU_UTILIZATION = Gauge(
    "xlit_gpu_utilization_percent",
    "GPU utilization percentage reported by nvidia-smi.",
    ["gpu", "name"],
)
GPU_MEMORY_USED_BYTES = Gauge(
    "xlit_gpu_memory_used_bytes",
    "GPU memory used in bytes reported by nvidia-smi.",
    ["gpu", "name"],
)
GPU_MEMORY_TOTAL_BYTES = Gauge(
    "xlit_gpu_memory_total_bytes",
    "Total GPU memory in bytes reported by nvidia-smi.",
    ["gpu", "name"],
)
GPU_TEMPERATURE_CELSIUS = Gauge(
    "xlit_gpu_temperature_celsius",
    "GPU temperature in Celsius reported by nvidia-smi.",
    ["gpu", "name"],
)
GPU_POWER_DRAW_WATTS = Gauge(
    "xlit_gpu_power_draw_watts",
    "GPU power draw in watts reported by nvidia-smi.",
    ["gpu", "name"],
)


def _endpoint_label():
    return request.endpoint or request.path


def _cpu_time_seconds():
    cpu_times = PROCESS.cpu_times()
    return cpu_times.user + cpu_times.system


def _collect_process_metrics():
    with PROCESS.oneshot():
        memory = PROCESS.memory_info()
        PROCESS_CPU_PERCENT.set(PROCESS.cpu_percent(interval=None))
        PROCESS_RSS_BYTES.set(memory.rss)
        PROCESS_VMS_BYTES.set(memory.vms)
        PROCESS_THREADS.set(PROCESS.num_threads())
        if hasattr(PROCESS, "num_fds"):
            PROCESS_OPEN_FDS.set(PROCESS.num_fds())


def _collect_gpu_metrics():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        gpu, name, utilization, memory_used, memory_total, temperature, power_draw = parts
        labels = {"gpu": gpu, "name": name}
        try:
            GPU_UTILIZATION.labels(**labels).set(float(utilization))
            GPU_MEMORY_USED_BYTES.labels(**labels).set(float(memory_used) * 1024 * 1024)
            GPU_MEMORY_TOTAL_BYTES.labels(**labels).set(float(memory_total) * 1024 * 1024)
            GPU_TEMPERATURE_CELSIUS.labels(**labels).set(float(temperature))
            GPU_POWER_DRAW_WATTS.labels(**labels).set(float(power_draw))
        except ValueError:
            continue


def _time_inference(direction, lang_code, mode, fn):
    start = time.perf_counter()
    try:
        return fn()
    finally:
        INFERENCE_LATENCY.labels(direction, lang_code, mode).observe(time.perf_counter() - start)


@app.before_request
def record_request_start():
    if request.endpoint == "metrics":
        return
    request.environ[REQUEST_START] = time.perf_counter()
    request.environ[REQUEST_CPU_START] = _cpu_time_seconds()
    IN_PROGRESS.inc()


@app.after_request
def record_request_metrics(response):
    if request.endpoint == "metrics":
        return response
    endpoint = _endpoint_label()
    REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
    start = request.environ.get(REQUEST_START)
    if start is not None:
        REQUEST_LATENCY.labels(request.method, endpoint).observe(time.perf_counter() - start)
    cpu_start = request.environ.get(REQUEST_CPU_START)
    if cpu_start is not None:
        REQUEST_CPU_TIME.labels(request.method, endpoint).observe(max(_cpu_time_seconds() - cpu_start, 0))
    IN_PROGRESS.dec()
    return response


@app.route("/metrics", methods=["GET"])
def metrics():
    _collect_process_metrics()
    _collect_gpu_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

## ----------------------------- Xlit Engine -------------------------------- ##

from .xlit_src import XlitEngine

MAX_SUGGESTIONS = 8
DEFAULT_NUM_SUGGESTIONS = 5

ENGINE = {
    "en2indic": XlitEngine(beam_width=MAX_SUGGESTIONS, rescore=False, model_type="transformer", src_script_type = "roman"),
    "indic2en": XlitEngine(beam_width=MAX_SUGGESTIONS, rescore=False, model_type="transformer", src_script_type = "indic"),
}

EXPOSED_LANGS = [
    {
        "LangCode": lang_code, # ISO-639 code
        "Identifier": lang_code, # ISO-639 code
        "DisplayName": LANG_CODE_TO_DISPLAY_NAME[lang_code],
        "Author": "AI4Bharat", # Name of developer / team
        "CompiledDate": "09-April-2022", # date on which model was trained
        "IsStable": True, # Set `False` if the model is experimental
        "Direction": "rtl" if lang_code in RTL_LANG_CODES else "ltr",
        "ScriptCode": LANG_CODE_TO_SCRIPT_CODE[lang_code],
        "GoogleFont": GOOGLE_FONTS[lang_code] if lang_code in GOOGLE_FONTS else None,
        "FallbackFont": FALLBACK_FONTS[lang_code] if lang_code in FALLBACK_FONTS else None,
    } for lang_code in sorted(ENGINE["en2indic"].all_supported_langs)
]

def get_app():
    return app, ENGINE

## ---------------------------- API End-points ------------------------------ ##

@app.route('/languages', methods = ['GET', 'POST'])
def supported_languages():
    # Format - https://xlit-api.ai4bharat.org/languages
    response = make_response(jsonify(EXPOSED_LANGS))
    if 'xlit_user_id' not in request.cookies:
        # host = request.environ['HTTP_ORIGIN'].split('://')[1]
        host = '.ai4bharat.org'
        response.set_cookie('xlit_user_id', uuid4().hex, max_age=365*24*60*60, domain=host, samesite='None', secure=True, httponly=True)
    return response

@app.route('/tl/<lang_code>/<eng_word>', methods = ['GET'])
@limiter.limit("5/second")
def xlit_api(lang_code, eng_word):
    # Format: https://xlit-api.ai4bharat.org/tl/ta/bharat
    response = {
        'success': False,
        'error': '',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': eng_word.strip(),
        'result': ''
    }

    transliterate_numerals = request.args.get('transliterate_numerals', default=False, type=lambda v: v.lower() == 'true')
    num_suggestions = request.args.get('num_suggestions', default=DEFAULT_NUM_SUGGESTIONS, type=int)

    if lang_code not in ENGINE["en2indic"].all_supported_langs:
        response['error'] = 'Invalid scheme identifier. Supported languages are: '+ str(ENGINE["en2indic"].all_supported_langs)
        return jsonify(response)

    try:
        ## Limit char count to --> 70
        xlit_result = _time_inference(
            "en2indic",
            lang_code,
            "word",
            lambda: ENGINE["en2indic"].translit_word(eng_word[:70], lang_code, topk=num_suggestions, transliterate_numerals=transliterate_numerals),
        )
    except Exception as e:
        xlit_result = XlitError.internal_err


    if isinstance(xlit_result, XlitError):
        response['error'] = xlit_result.value
        print("XlitError:", traceback.format_exc())
    else:
        response['result'] = xlit_result
        response['success'] = True

    return jsonify(response)

@app.route('/rtl/<lang_code>/<word>', methods = ['GET'])
@limiter.limit("5/second")
def reverse_xlit_api(lang_code, word):
    # Format: https://api.varnamproject.com/rtl/hi/भारत
    response = {
        'success': False,
        'error': '',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': word.strip(),
        'result': ''
    }

    if lang_code not in ENGINE["indic2en"].all_supported_langs:
        response['error'] = 'Invalid scheme identifier. Supported languages are: '+ str(ENGINE["indic2en"].all_supported_langs)
        return jsonify(response)

    num_suggestions = request.args.get('num_suggestions', default=DEFAULT_NUM_SUGGESTIONS, type=int)

    try:
        ## Limit char count to --> 70
        xlit_result = _time_inference(
            "indic2en",
            lang_code,
            "word",
            lambda: ENGINE["indic2en"].translit_word(word[:70], lang_code, topk=num_suggestions),
        )
    except Exception as e:
        xlit_result = XlitError.internal_err

    if isinstance(xlit_result, XlitError):
        response['error'] = xlit_result.value
        print("XlitError:", traceback.format_exc())
    else:
        response['result'] = xlit_result
        response['success'] = True

    return jsonify(response)

@app.route('/transliterate', methods=['POST'])
@limiter.limit("5/second")
def ulca_api():
    '''
    ULCA-compliant endpoint. See for sample request-response:
    https://github.com/ULCA-IN/ulca/tree/master/specs/examples/model/transliteration-model
    '''
    data = request.get_json(force=True)
    
    if "input" not in data or "config" not in data:
        return jsonify({
            "status": {
                "statusCode": 400,
                "message": "Ensure `input` and `config` fields missing."
            }
        }), 400
    
    if (data["config"]["language"]["sourceLanguage"] == "en" and data["config"]["language"]["targetLanguage"] in ENGINE["en2indic"].all_supported_langs) or (data["config"]["language"]["sourceLanguage"] in ENGINE["indic2en"].all_supported_langs and data["config"]["language"]["targetLanguage"] == 'en'):
        pass
    else:
        return jsonify({
            "status": {
                "statusCode": 501,
                "message": "The mentioned language-pair is not supported yet."
            }
        }), 501
    
    is_sentence = data["config"]["isSentence"] if "isSentence" in data["config"] else False
    num_suggestions = 1 if is_sentence else (data["config"]["numSuggestions"] if "numSuggestions" in data["config"] else 5)

    if data["config"]["language"]["targetLanguage"] == "en":
        engine = ENGINE["indic2en"]
        lang_code = data["config"]["language"]["sourceLanguage"]
    else:
        engine = ENGINE["en2indic"]
        lang_code = data["config"]["language"]["targetLanguage"]

    outputs = []
    for item in data["input"]:
        if is_sentence:
            item["target"] = [_time_inference(
                "indic2en" if data["config"]["language"]["targetLanguage"] == "en" else "en2indic",
                lang_code,
                "sentence",
                lambda source=item["source"]: engine.translit_sentence(source, lang_code=lang_code),
            )]
        else:
            item["source"] = item["source"][:32]
            item["target"] = _time_inference(
                "indic2en" if data["config"]["language"]["targetLanguage"] == "en" else "en2indic",
                lang_code,
                "word",
                lambda source=item["source"]: engine.translit_word(source, lang_code=lang_code, topk=num_suggestions),
            )
    
    return {
        "output": data["input"],
        # "status": {
        #     "statusCode": 200,
        #     "message" : "success"
        # }
    }, 200
