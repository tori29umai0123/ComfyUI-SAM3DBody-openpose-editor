"""Hugging Face downloads with a scoped IPv4 fallback for WinError 10054."""

import json
import os
import subprocess
import sys
import time


_RESULT_PREFIX = "SAM3DBODY_DOWNLOAD_RESULT:"


def _is_winerror_10054(error):
    if sys.platform != "win32":
        return False
    seen = set()
    pending = [error]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if getattr(current, "winerror", None) == 10054 or "10054" in str(current):
            return True
        pending.extend(cause for cause in (current.__cause__, current.__context__) if cause is not None)
    return False


def download(kind, **kwargs):
    """Use the normal Hub client first; retry 10054 in an isolated IPv4 process."""
    from huggingface_hub import hf_hub_download, snapshot_download

    if kind not in ("file", "snapshot"):
        raise ValueError(f"Unsupported download kind: {kind}")
    kwargs.setdefault("etag_timeout", 30)
    if kind == "snapshot":
        kwargs.setdefault("max_workers", 1)
    operation = hf_hub_download if kind == "file" else snapshot_download
    try:
        return operation(**kwargs)
    except Exception as error:
        if not _is_winerror_10054(error):
            raise
        print(f"[SAM3DBodyOpenPose] Hugging Face download for {kwargs.get('repo_id')} hit WinError 10054; retrying over IPv4.")

    payload = json.dumps({"kind": kind, "kwargs": kwargs})
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    last_error = None
    for attempt in range(4):
        if attempt:
            time.sleep(2 ** attempt)
        result = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--ipv4", payload],
            capture_output=True, text=True, env=environment, check=False,
        )
        lines = result.stdout.splitlines()
        if result.returncode == 0 and lines and lines[-1].startswith(_RESULT_PREFIX):
            return lines[-1][len(_RESULT_PREFIX):]
        last_error = (result.stdout + "\n" + result.stderr).strip()[-2000:]
        print(f"[SAM3DBodyOpenPose] IPv4 download attempt {attempt + 1}/4 failed for {kwargs.get('repo_id')}.")
    raise RuntimeError(f"Hugging Face download failed for {kwargs.get('repo_id')} after IPv4 retries: {last_error}")


def _run_ipv4(payload):
    import httpx
    from huggingface_hub import hf_hub_download, set_client_factory, snapshot_download

    # This process exists solely for the retry, so the factory cannot affect ComfyUI.
    set_client_factory(lambda: httpx.Client(
        transport=httpx.HTTPTransport(local_address="0.0.0.0"),
        follow_redirects=True, timeout=60,
    ))
    request = json.loads(payload)
    operation = hf_hub_download if request["kind"] == "file" else snapshot_download
    path = operation(**request["kwargs"])
    print(_RESULT_PREFIX + os.fspath(path))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--ipv4":
        raise SystemExit("Usage: model_download.py --ipv4 JSON_PAYLOAD")
    _run_ipv4(sys.argv[2])
