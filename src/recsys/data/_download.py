"""Shared download primitives for dataset loaders.

Two functions live here:

* :func:`http_download_atomic` — stream a URL to disk via a ``.part``
  file with Content-Length verification and an atomic rename on
  success. Used by the Zenodo-backed loaders (KuaiRec, KuaiRand).
* :func:`fetch_hf_dataset` — snapshot-download a Hugging Face Hub
  dataset repo into a local cache directory. Thin wrapper around
  ``huggingface_hub.snapshot_download``.

Neither function is registered anywhere — they're plumbing that dataset
loaders import directly.
"""

from __future__ import annotations

import logging
import sys
import urllib.request
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": "curl/8.4.0",
    "Accept": "*/*",
}


def http_download_atomic(
    url: str,
    dest: Path,
    *,
    chunk_size: int = 1 << 20,
    headers: dict[str, str] | None = None,
    verify_content_length: bool = True,
    progress_label: str = "download",
    progress_every_bytes: int = 16 * (1 << 20),
    timeout_s: float = 60.0,
) -> None:
    """Stream ``url`` to ``dest`` atomically.

    Writes to ``dest.with_suffix(dest.suffix + '.part')`` first and
    calls ``.replace(dest)`` on success, so interrupted downloads never
    leave a truncated file where the cached one should be. Progress is
    logged to stderr roughly every ``progress_every_bytes`` bytes.

    When ``verify_content_length`` is True and the server sends a
    ``Content-Length`` header, a final size check raises
    ``RuntimeError`` rather than renaming a short file into place.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    req_headers = headers if headers is not None else _DEFAULT_HEADERS
    req = urllib.request.Request(url, headers=req_headers)
    LOGGER.info("%s: downloading %s -> %s", progress_label, url, dest)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total else None
            downloaded = 0
            last_report = 0
            with part.open("wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_report >= progress_every_bytes:
                        last_report = downloaded
                        if total_bytes:
                            pct = 100.0 * downloaded / total_bytes
                            msg = (
                                f"  {progress_label}: {downloaded/1e6:.1f}/"
                                f"{total_bytes/1e6:.1f} MB ({pct:5.1f}%)"
                            )
                        else:
                            msg = f"  {progress_label}: {downloaded/1e6:.1f} MB"
                        print(msg, file=sys.stderr, flush=True)
    except BaseException:
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass
        raise

    if (
        verify_content_length
        and total_bytes is not None
        and downloaded != total_bytes
    ):
        try:
            part.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"{progress_label} truncated: got {downloaded} bytes, "
            f"expected {total_bytes} (url={url})"
        )
    part.replace(dest)


def fetch_hf_dataset(
    repo_id: str,
    cache_dir: Path | str,
    *,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
    token: str | None = None,
) -> Path:
    """Snapshot-download a Hugging Face Hub dataset repo.

    Thin wrapper around ``huggingface_hub.snapshot_download`` that pins
    the repo_type to ``dataset`` and returns the local snapshot path.
    The snapshot is cached under ``cache_dir``; subsequent calls with
    the same ``repo_id`` (and ``revision``) hit the cache.

    Parameters
    ----------
    repo_id:
        Hub repo id, e.g. ``"TAAC2026/data_sample_1000"``.
    cache_dir:
        Local cache directory. Created if missing.
    revision:
        Optional git revision / branch / tag.
    allow_patterns:
        Optional glob list forwarded to ``snapshot_download`` to limit
        which files are fetched (useful for large repos where only a
        subset of shards is needed).
    token:
        Optional Hugging Face access token for private / gated repos.

    Returns
    -------
    Path to the on-disk snapshot directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "fetch_hf_dataset requires the 'huggingface_hub' package. "
            "Install it via `uv add huggingface_hub`."
        ) from e

    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info("HF Hub: fetching %s -> %s", repo_id, cache_path)
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        cache_dir=str(cache_path),
        revision=revision,
        allow_patterns=allow_patterns,
        token=token,
    )
    return Path(local_path)


__all__ = ["http_download_atomic", "fetch_hf_dataset"]
