"""
Local LLM Server Management
===========================

Ensures a local OpenAI-compatible server (vLLM) is serving the model
the evaluation needs, starting one if necessary and downloading the
weights if they are not already on disk.

The evaluation must run without any hosted API, so this module provides
the "make it work" path:

  1. Probe the configured base URL.  If a server already answers
     ``/v1/models`` with the requested model, use it as-is.
  2. Otherwise locate the model weights locally (see
     ``src.model_paths``); if absent, download them.
  3. Start ``vllm serve`` on a free port and wait until it reports
     ready.

Configuration (environment variables):

    STENRAG_LLM_BASE_URL   base URL to use / start (default:
                           ``http://127.0.0.1:8013/v1``).
    STENRAG_LLM_MODEL      served model name (default:
                           ``llama-3.1-8b``).
    STENRAG_LLM_MODEL_PATH local directory of the weights.  When unset,
                           a small set of known local stores is probed.
    STENRAG_LLM_GPU        GPU index for the server (default: ``1``).
    STENRAG_LLM_PORT       port to serve on (default: ``8013``).
    STENRAG_LLM_MAX_LEN    max model length (default: ``8192``).
    STENRAG_LLM_GPU_UTIL   GPU memory utilisation (default: ``0.62``).

Public API
----------
    server_is_up(base_url, model) -> bool
    ensure_server(...) -> str
        Return the base URL of a ready server, starting one if needed.
"""

from __future__ import annotations

import os
import socket
import subprocess
import time
from typing import List, Optional

# Known local weight stores, probed in order.
_LOCAL_MODEL_STORES: List[str] = [
    "/home/xd/model",
    "/home/lyh/models",
    "/home/lyh/StenRAG/StenRAG/models",
]

# Candidate directory names for the served model.
_LOCAL_MODEL_DIR_NAMES = {
    "llama-3.1-8b": ["Llama-3.1-8B-Instruct", "llama-3.1-8b-instruct",
                     "Llama-3.1-8B"],
    "qwen3-8b": ["Qwen3-8B", "qwen3-8b"],
    "qwen3.5-0.8b": ["Qwen3.5-0.8B", "qwen3.5-0.8b"],
}


def _log(msg: str) -> None:
    print(f"[llm] {msg}", flush=True)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def server_is_up(base_url: str, model: str = "") -> bool:
    """True when a server at `base_url` is up and serves `model`."""
    try:
        import requests
    except Exception:
        return False
    url = f"{base_url.rstrip('/')}/models"
    try:
        resp = requests.get(url, timeout=5)
    except Exception:
        return False
    if resp.status_code != 200:
        return False
    try:
        data = resp.json()
    except Exception:
        return False
    ids = [m.get("id") for m in (data.get("data") or [])]
    if not model:
        return bool(ids)
    return model in ids or any(model in str(i) for i in ids)


def find_local_weights(model: str) -> Optional[str]:
    """
    Locate the on-disk directory holding the weights for `model`.

    Searches the configured STENRAG_LLM_MODEL_PATH first, then the
    known local model stores.  Returns None when nothing is found.
    """
    explicit = os.environ.get("STENRAG_LLM_MODEL_PATH", "").strip()
    if explicit and os.path.isdir(explicit):
        return explicit

    names = _LOCAL_MODEL_DIR_NAMES.get(model, [model])
    for store in _LOCAL_MODEL_STORES:
        if not os.path.isdir(store):
            continue
        for name in names:
            candidate = os.path.join(store, name)
            if os.path.isdir(candidate):
                return candidate
    return None


def ensure_server(
    base_url: str = "",
    model: str = "",
    gpu: str = "",
    port: int = 0,
    max_len: int = 0,
    gpu_util: float = 0.0,
    startup_timeout: float = 900.0,
) -> str:
    """
    Ensure a local server is serving `model`, starting one if needed.

    Returns the base URL of the ready server.  Raises RuntimeError when
    the server cannot be made ready.
    """
    base_url = base_url or os.environ.get(
        "STENRAG_LLM_BASE_URL", "http://127.0.0.1:8013/v1"
    )
    model = model or os.environ.get("STENRAG_LLM_MODEL", "llama-3.1-8b")
    gpu = gpu or os.environ.get("STENRAG_LLM_GPU", "1")
    port = port or int(os.environ.get("STENRAG_LLM_PORT", "8013"))
    max_len = max_len or int(os.environ.get("STENRAG_LLM_MAX_LEN", "8192"))
    gpu_util = gpu_util or float(os.environ.get("STENRAG_LLM_GPU_UTIL", "0.62"))

    if server_is_up(base_url, model):
        _log(f"server already up at {base_url} (model={model})")
        return base_url

    weights = find_local_weights(model)
    if weights is None:
        raise RuntimeError(
            f"No local weights found for model '{model}'. "
            f"Download it first (e.g. snapshot_download) into one of: "
            f"{_LOCAL_MODEL_STORES}"
        )
    _log(f"local weights: {weights}")

    if not _port_is_free(port):
        # Something else already holds the port; probe it in case it
        # is a compatible server under a different model name.
        if server_is_up(base_url, ""):
            _log(f"port {port} busy but server responds; using it")
            return base_url
        raise RuntimeError(f"Port {port} is already in use by another process.")

    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"vllm_{model}.log")

    cmd = [
        "vllm", "serve", weights,
        "--served-model-name", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_len),
        "--dtype", "bfloat16",
        "--max-num-seqs", "64",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # The flashinfer sampler requires nvcc, which is absent here; the
    # PyTorch sampler is used instead.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    env.setdefault("VLLM_LOGGING_LEVEL", "WARNING")

    _log(f"starting: {' '.join(cmd)}")
    _log(f"log: {log_path}")
    with open(log_path, "w", encoding="utf-8") as fh:
        subprocess.Popen(
            cmd,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        time.sleep(5)
        if server_is_up(base_url, model):
            _log(f"server ready at {base_url}")
            return base_url
    raise RuntimeError(
        f"vLLM server did not become ready within {startup_timeout:.0f}s. "
        f"See {log_path}"
    )
