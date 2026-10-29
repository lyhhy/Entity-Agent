"""
Local Model Path Resolution
===========================

Resolves HuggingFace model *names* (e.g. ``"all-MiniLM-L6-v2"``) to
on-disk directories so that every component loads the model from the
local filesystem instead of hitting the Hub.

Why this exists
---------------
The evaluation must run fully offline: the Hub is unreachable from this
host (verified: ``Network is unreachable``) and the Xet-backed mirror
rejects anonymous reconstruction (``401 Unauthorized``).  Models were
therefore pre-downloaded with ``snapshot_download`` into a local
directory tree, and this module maps names onto that tree.

Layout
------
The local store follows the HF repo naming convention, with ``/``
replaced by ``--``::

    <model_dir>/all-MiniLM-L6-v2/
    <model_dir>/ms-marco-MiniLM-L-6-v2/

so the two on-disk names are derived from the HF repo ids
``sentence-transformers/all-MiniLM-L6-v2`` and
``cross-encoder/ms-marco-MiniLM-L-6-v2``.

Configuration
-------------
``STENRAG_MODEL_DIR`` overrides the root directory (default:
``/home/lyh/models``).

Public API
----------
    resolve_model(name) -> str
        Return a local path when the model exists locally, otherwise the
        original name (so that the normal HF resolution path still runs
        for models that were never downloaded).
    ensure_model(name) -> str
        Return a local path, downloading the model into the local store
        first if it is not there yet.  Falls back to the original name
        when the download cannot be performed (e.g. no network).
"""

from __future__ import annotations

import os
from typing import Dict

# Root of the local model store.
DEFAULT_MODEL_DIR = "/home/lyh/models"

# Canonical repo id -> local directory name.  The local name mirrors the
# HF repo id with "/" replaced by "--" (see module docstring).
_REPO_TO_LOCAL: Dict[str, str] = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "ms-marco-MiniLM-L-6-v2",
}

# Short aliases used throughout the codebase (the `semantic_embed_model`
# / `embedder_model` / `cross_encoder_model_name` defaults).
_ALIAS_TO_REPO: Dict[str, str] = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
}


def model_dir() -> str:
    """Return the root directory of the local model store."""
    return os.environ.get("STENRAG_MODEL_DIR", DEFAULT_MODEL_DIR)


def _has_weights(directory: str) -> bool:
    """True when `directory` looks like a usable model directory.

    Requires at least one weight file so that a half-written or
    empty directory (e.g. an interrupted download) is not mistaken for
    a valid model.
    """
    if not os.path.isdir(directory):
        return False
    try:
        entries = os.listdir(directory)
    except OSError:
        return False
    for name in entries:
        if name.endswith(".safetensors") or name.endswith(".bin"):
            return True
    return False


def resolve_model(name: str) -> str:
    """
    Map a model name (or HF repo id) to a local directory path.

    Returns the local path when the model is present in the local store;
    otherwise returns `name` unchanged, preserving the upstream
    behaviour of resolving through the Hub.
    """
    if not name:
        return name

    key = name.strip()
    repo = _ALIAS_TO_REPO.get(key) or _ALIAS_TO_REPO.get(key.lower())
    if repo is None and "/" in key:
        repo = key

    if repo is not None:
        local_name = _REPO_TO_LOCAL.get(repo)
        if local_name is None:
            # Unknown repo: derive the local name from the repo id.
            local_name = repo.replace("/", "--")
        candidate = os.path.join(model_dir(), local_name)
        if _has_weights(candidate):
            return candidate

    return name


def _repo_for(name: str):
    """Return the HF repo id for `name`, or None if it cannot be mapped."""
    key = (name or "").strip()
    repo = _ALIAS_TO_REPO.get(key) or _ALIAS_TO_REPO.get(key.lower())
    if repo is None and "/" in key:
        repo = key
    return repo


def ensure_model(name: str) -> str:
    """
    Return a local path for `name`, downloading it into the local store
    if it is not already there.

    The download uses `huggingface_hub.snapshot_download` with Xet
    disabled (the Xet reconstruction endpoint rejects anonymous
    requests on this host with HTTP 401) and the mirror endpoint when
    one is reachable.

    If the download fails for any reason, the original `name` is
    returned so that the caller's normal HuggingFace resolution path
    still applies.  This function never raises.
    """
    resolved = resolve_model(name)
    if resolved != name:
        # Already present locally.
        return resolved

    repo = _repo_for(name)
    if repo is None:
        return name

    local_name = _REPO_TO_LOCAL.get(repo) or repo.replace("/", "--")
    target = os.path.join(model_dir(), local_name)
    if _has_weights(target):
        return target

    try:
        from huggingface_hub import snapshot_download
    except Exception:
        return name

    # The Xet backend is unusable here (HTTP 401 on reconstruction) and
    # the default endpoint is unreachable; prefer the mirror with Xet off.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        snapshot_download(repo, local_dir=target)
    except Exception:
        return name

    return target if _has_weights(target) else name
