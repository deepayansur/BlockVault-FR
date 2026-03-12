from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Sequence

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eval.cpu_benchmark import benchmark_local_verify, benchmark_worker_topologies
from eval.pipeline import build_item_paths, collect_class_images, sample_items_balanced
from eval.vggface2_verify import run_vggface2_verification

DEFAULT_VGG_ROOT = os.path.join(ROOT, "datasets", "VGGface2_HQ_1", "VGGface2_None_norm_512_true_bygfpgan")


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _no_face_rate(summary: Dict[str, Any]) -> float:
    total = max(1, int(summary["coverage"]["total_images"]))
    return float(summary["coverage"]["no_face"] / total)


def _tar(summary: Dict[str, Any], far_key: str) -> float:
    return float(summary["verification"]["far_targets"][far_key]["tar"])


def _guardrail_ok(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    eer_delta = candidate["verification"]["eer"]["value"] - baseline["verification"]["eer"]["value"]
    tar_delta = _tar(baseline, "1e-04") - _tar(candidate, "1e-04")
    no_face_delta = _no_face_rate(candidate) - _no_face_rate(baseline)
    return eer_delta <= 0.01 and tar_delta <= 0.03 and no_face_delta <= 0.02


def _benchmark_image_paths(vgg_root: str, max_images: int, seed: int, limit: int = 24) -> List[str]:
    sampled = sample_items_balanced(collect_class_images(vgg_root), max_images=max_images, seed=seed)
    item_paths = build_item_paths(sampled, vgg_root)
    return [path for _, path in item_paths[:limit]]


def _select_vgg_config(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    baseline = next((candidate for candidate in candidates if candidate["config"]["det_size"] == 640), candidates[0])
    eligible = [candidate for candidate in candidates if _guardrail_ok(candidate, baseline)]
    pool = eligible or list(candidates)
    return sorted(pool, key=lambda candidate: (candidate["latency"]["local"]["wall_ms"]["p95"], candidate["config"]["det_size"]))[0]


def _build_text_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("CPU Face Verification Results")
    lines.append("")
    lines.append("Run configuration")
    lines.append(f"  Stage: {report['run']['stage']}")
    lines.append(f"  VGG root: {report['run']['vgg_root']}")
    lines.append(f"  Max images: {report['run']['max_images']}")
    lines.append(f"  Num workers: {report['run']['num_workers']}")
    lines.append("")
    lines.append("Detector sweep")
    for candidate in report["detector_sweep"]:
        lines.append(
            f"  det_size={candidate['config']['det_size']} backend={candidate['parallel_backend']} "
            f"p95(local)={candidate['latency']['local']['wall_ms']['p95']:.2f}ms "
            f"EER={candidate['verification']['eer']['value']:.4f} TAR@1e-4={candidate['verification']['far_targets']['1e-04']['tar']:.4f} "
            f"no_face={candidate['coverage']['no_face']}"
        )

    failures = report.get("failed_detector_sweep") or []
    if failures:
        lines.append("")
        lines.append("Skipped detector sizes")
        for failure in failures:
            lines.append(f"  det_size={failure['det_size']} error={failure['error']}")
    lines.append("")

    selected = report["selected_config"]
    lines.append("Selected CPU config")
    lines.append(f"  det_size: {selected['config']['det_size']}")
    lines.append(f"  threshold: {_format_optional(selected['authorization']['threshold'])}")
    lines.append(f"  selected workers: {report['worker_topologies']['selected']['workers']}")
    lines.append("")

    summary = report["vggface2"]
    coverage = summary["coverage"]
    verification = summary["verification"]
    authorization = summary["authorization"]
    lines.append(f"{summary['dataset']} metrics")
    lines.append(f"  Parallel backend: {summary.get('parallel_backend', 'serial')}")
    lines.append(f"  Total images: {coverage['total_images']}")
    lines.append(f"  Classes: {coverage['classes']}")
    lines.append(f"  Embedded images: {coverage['embedded_images']}")
    lines.append(f"  Skipped images: {coverage['skipped_images']}")
    lines.append(f"  read_fail={coverage['read_fail']} no_face={coverage['no_face']}")
    lines.append(f"  ROC-AUC: {verification['roc_auc']:.4f}")
    lines.append(f"  EER: {verification['eer']['value']:.4f} @ threshold {verification['eer']['threshold']:.4f}")
    for key, values in verification["far_targets"].items():
        lines.append(f"  TAR@FAR={key}: {values['tar']:.4f} @ threshold {values['threshold']:.4f}")
    lines.append(f"  Rank-1: {_format_optional(authorization['rank1'])}")
    lines.append(f"  Enrolled accept rate: {_format_optional(authorization['enrolled_accept_rate'])}")
    lines.append(f"  Unknown reject rate: {_format_optional(authorization['unknown_reject_rate'])}")
    lines.append("")

    lines.append("Latency")
    local_latency = report["latency"]["local"]
    lines.append(
        f"  Local verify: p50={local_latency['wall_ms']['p50']:.2f}ms p95={local_latency['wall_ms']['p95']:.2f}ms p99={local_latency['wall_ms']['p99']:.2f}ms"
    )
    http_selected = report["worker_topologies"]["selected"]["four_client"]
    lines.append(
        f"  HTTP /verify (4 clients, workers={report['worker_topologies']['selected']['workers']}): "
        f"p50={http_selected['wall_ms']['p50']:.2f}ms p95={http_selected['wall_ms']['p95']:.2f}ms p99={http_selected['wall_ms']['p99']:.2f}ms"
    )
    lines.append("")

    lines.append("Worker topology sweep")
    for profile in report["worker_topologies"]["profiles"]:
        four_client = profile["four_client"]
        single = profile["single_client"]
        lines.append(
            f"  workers={profile['workers']} single_p95={single['wall_ms']['p95']:.2f}ms "
            f"four_client_p95={four_client['wall_ms']['p95']:.2f}ms four_client_rps={four_client['throughput_rps']:.2f}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vgg-root", default=DEFAULT_VGG_ROOT)
    ap.add_argument("--det-sizes", default="320,384,448,640")
    ap.add_argument("--max-images", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--bootstrap", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--local-repeats", type=int, default=2)
    ap.add_argument("--http-repeats", type=int, default=2)
    ap.add_argument("--results-json", default=os.path.join(ROOT, "results.json"))
    ap.add_argument("--results-txt", default=os.path.join(ROOT, "results.txt"))
    args = ap.parse_args()

    det_sizes = [int(value.strip()) for value in args.det_sizes.split(",") if value.strip()]
    benchmark_paths = _benchmark_image_paths(args.vgg_root, max_images=args.max_images, seed=args.seed)

    candidates: List[Dict[str, Any]] = []
    failed_candidates: List[Dict[str, Any]] = []
    for det_size in det_sizes:
        cache_path = os.path.join(ROOT, "eval", f"cache_vgg_det{det_size}_{args.max_images}.npz")
        try:
            summary = run_vggface2_verification(
                root=args.vgg_root,
                det_size=det_size,
                max_images=args.max_images,
                cache_path=cache_path,
                seed=args.seed,
                num_workers=args.num_workers,
                bootstrap=args.bootstrap,
                use_gpu=False,
            )
            summary["latency"] = {
                "local": benchmark_local_verify(benchmark_paths, det_size=det_size, repeats=args.local_repeats),
            }
            candidates.append(summary)
        except RuntimeError as exc:
            failed_candidates.append({"det_size": int(det_size), "error": str(exc)})

    if not candidates:
        details = "; ".join(
            f"det_size={failure['det_size']}: {failure['error']}" for failure in failed_candidates
        ) or "no successful detector sizes"
        raise RuntimeError(f"All detector-size candidates failed. {details}")

    selected = _select_vgg_config(candidates)
    worker_topologies = benchmark_worker_topologies(
        benchmark_paths,
        det_size=selected["config"]["det_size"],
        repeats=args.http_repeats,
    )

    report = {
        "run": {
            "stage": f"stage1_cpu_tuning_{args.max_images}",
            "vgg_root": os.path.abspath(args.vgg_root),
            "max_images": int(args.max_images),
            "num_workers": int(args.num_workers),
            "bootstrap": int(args.bootstrap),
            "seed": int(args.seed),
        },
        "detector_sweep": candidates,
        "failed_detector_sweep": failed_candidates,
        "selected_config": selected,
        "vggface2": selected,
        "latency": {
            "local": selected["latency"]["local"],
        },
        "worker_topologies": worker_topologies,
    }

    with open(args.results_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(args.results_txt, "w", encoding="utf-8") as handle:
        handle.write(_build_text_report(report))

    print(f"Wrote {args.results_json}")
    print(f"Wrote {args.results_txt}")


if __name__ == "__main__":
    main()
