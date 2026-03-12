from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Sequence

import requests

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from app import encode_jpeg_dataurl, resize_max_side
from eval.pipeline import prepare_image
from face_service.ml import FaceEngine

DEFAULT_BENCHMARK_PORT = 8011


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lower = int(pos)
    upper = min(len(ordered) - 1, lower + 1)
    weight = pos - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "p50": float(_quantile(values, 0.50)),
        "p95": float(_quantile(values, 0.95)),
        "p99": float(_quantile(values, 0.99)),
        "mean": float(sum(values) / len(values)),
    }


def _timing_percentiles(results: Sequence[Dict[str, Any]], key: str) -> Dict[str, float]:
    values: List[float] = []
    for result in results:
        timings = result.get("timingsMs") or {}
        value = timings.get(key)
        if value is not None:
            values.append(float(value))
    return _percentiles(values)


def _load_benchmark_images(image_paths: Sequence[str], max_side: int, jpeg_quality: int) -> List[Dict[str, Any]]:
    payloads: List[Dict[str, Any]] = []
    for path in image_paths:
        img = prepare_image(path)
        if img is None:
            continue
        resized = resize_max_side(img, max_side)
        payloads.append({"path": path, "img": resized, "dataurl": encode_jpeg_dataurl(resized, q=jpeg_quality)})
    return payloads


def benchmark_local_verify(image_paths: Sequence[str], det_size: int, repeats: int = 2, warmup: int = 4) -> Dict[str, Any]:
    payloads = _load_benchmark_images(image_paths, max_side=960, jpeg_quality=75)
    if not payloads:
        raise RuntimeError("No benchmark images could be prepared.")

    engine = FaceEngine(det_size=det_size, use_gpu=False)
    engine.init_models()
    engine.load_db()

    for idx in range(min(warmup, len(payloads))):
        engine.verify_bgr(payloads[idx]["img"])

    wall_times: List[float] = []
    results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for _ in range(max(1, repeats)):
        for payload in payloads:
            t0 = time.perf_counter()
            result = engine.verify_bgr(payload["img"])
            wall_times.append((time.perf_counter() - t0) * 1000.0)
            results.append(result)
    elapsed = max(time.perf_counter() - started, 1e-9)

    return {
        "mode": "local",
        "requests": int(len(results)),
        "throughput_rps": float(len(results) / elapsed),
        "wall_ms": _percentiles(wall_times),
        "timings_ms": {
            "decode": _timing_percentiles(results, "decode"),
            "infer": _timing_percentiles(results, "infer"),
            "match": _timing_percentiles(results, "match"),
            "total": _timing_percentiles(results, "total"),
        },
    }


def _cluster_env(det_size: int, workers: int) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FACE_USE_GPU": "0",
            "DET_SIZE": str(det_size),
            "FACE_RUNTIME_PROFILE": f"cpu-workers-{workers}",
            "FACE_SERVER_WORKERS": str(workers),
            "FACE_CPU_TARGET": "concurrent_throughput",
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "6"),
            "OMP_WAIT_POLICY": env.get("OMP_WAIT_POLICY", "PASSIVE"),
            "OMP_PROC_BIND": env.get("OMP_PROC_BIND", "TRUE"),
            "OMP_PLACES": env.get("OMP_PLACES", "cores"),
        }
    )
    return env


def _launch_single_instance(port: int, det_size: int, workers: int) -> subprocess.Popen[Any]:
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "face_service.app:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
        cwd=ROOT,
        env=_cluster_env(det_size, workers),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def _stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


@contextmanager
def launch_face_service_cluster(workers: int, det_size: int, base_port: int = DEFAULT_BENCHMARK_PORT) -> Any:
    processes: List[subprocess.Popen[Any]] = []
    ports = [base_port + idx for idx in range(max(1, workers))]
    try:
        for port in ports:
            processes.append(_launch_single_instance(port=port, det_size=det_size, workers=workers))

        base_urls = [f"http://127.0.0.1:{port}" for port in ports]
        health_payloads: List[Dict[str, Any]] = []
        deadline = time.time() + 90.0
        for base_url in base_urls:
            while time.time() < deadline:
                try:
                    response = requests.get(base_url + "/health", timeout=2)
                    response.raise_for_status()
                    health_payloads.append(response.json())
                    break
                except Exception:
                    time.sleep(1.0)
            else:
                raise RuntimeError(f"Timed out waiting for face_service on {base_url}")

        yield {"base_urls": base_urls, "health": health_payloads[0], "health_instances": health_payloads}
    finally:
        for process in processes:
            _stop_process(process)


def benchmark_http_verify(
    image_paths: Sequence[str],
    base_urls: Sequence[str] | str,
    concurrency: int,
    repeats: int = 2,
    warmup: int = 4,
    max_side: int = 960,
    jpeg_quality: int = 75,
) -> Dict[str, Any]:
    endpoints = [base_urls] if isinstance(base_urls, str) else list(base_urls)
    payloads = _load_benchmark_images(image_paths, max_side=max_side, jpeg_quality=jpeg_quality)
    if not payloads:
        raise RuntimeError("No benchmark images could be prepared.")

    requests_payloads = [payload["dataurl"] for payload in payloads] * max(1, repeats)

    for idx in range(min(warmup, len(requests_payloads))):
        base_url = endpoints[idx % len(endpoints)]
        response = requests.post(base_url + "/verify", json={"imageBase64": requests_payloads[idx]}, timeout=30)
        response.raise_for_status()

    def _send(base_url: str, dataurl: str) -> Dict[str, Any]:
        started = time.perf_counter()
        response = requests.post(base_url + "/verify", json={"imageBase64": dataurl}, timeout=60)
        response.raise_for_status()
        payload = response.json()
        payload["wall_ms"] = (time.perf_counter() - started) * 1000.0
        return payload

    wall_times: List[float] = []
    results: List[Dict[str, Any]] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = []
        for idx, payload in enumerate(requests_payloads):
            base_url = endpoints[idx % len(endpoints)]
            futures.append(executor.submit(_send, base_url, payload))
        for future in as_completed(futures):
            result = future.result()
            wall_times.append(float(result["wall_ms"]))
            results.append(result)
    elapsed = max(time.perf_counter() - started, 1e-9)

    return {
        "mode": "http",
        "instances": int(len(endpoints)),
        "concurrency": int(concurrency),
        "requests": int(len(results)),
        "throughput_rps": float(len(results) / elapsed),
        "wall_ms": _percentiles(wall_times),
        "timings_ms": {
            "decode": _timing_percentiles(results, "decode"),
            "infer": _timing_percentiles(results, "infer"),
            "match": _timing_percentiles(results, "match"),
            "total": _timing_percentiles(results, "total"),
        },
    }


def benchmark_worker_topologies(
    image_paths: Sequence[str],
    det_size: int,
    workers_list: Iterable[int] = (1, 2, 3),
    repeats: int = 2,
    port: int = DEFAULT_BENCHMARK_PORT,
) -> Dict[str, Any]:
    profiles: List[Dict[str, Any]] = []
    next_port = port
    for workers in workers_list:
        with launch_face_service_cluster(workers=workers, det_size=det_size, base_port=next_port) as service:
            single = benchmark_http_verify(image_paths, service["base_urls"], concurrency=1, repeats=repeats)
            concurrent = benchmark_http_verify(image_paths, service["base_urls"], concurrency=4, repeats=repeats)
            profiles.append(
                {
                    "workers": int(workers),
                    "health": service["health"],
                    "health_instances": service["health_instances"],
                    "single_client": single,
                    "four_client": concurrent,
                }
            )
        next_port += max(1, workers) + 2

    best_single_p95 = min(profile["single_client"]["wall_ms"]["p95"] for profile in profiles)
    eligible = [profile for profile in profiles if profile["four_client"]["wall_ms"]["p95"] <= (best_single_p95 * 1.15)]
    selection_pool = eligible or profiles
    selected = sorted(
        selection_pool,
        key=lambda profile: (-profile["four_client"]["throughput_rps"], profile["four_client"]["wall_ms"]["p95"], profile["workers"]),
    )[0]

    return {
        "profiles": profiles,
        "selected": {
            "workers": int(selected["workers"]),
            "single_client": selected["single_client"],
            "four_client": selected["four_client"],
            "health": selected["health"],
        },
        "selection_rule": "highest four-client throughput with no more than 15% p95 regression vs best single-client p95",
        "best_single_client_p95_ms": float(best_single_p95),
    }
