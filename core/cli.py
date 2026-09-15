import argparse
import json
import logging

from scripts.benchmark_registry import BENCHMARKS


def main(default_benchmark=None):
    parser = argparse.ArgumentParser(description="4DEvalKit: predicted 3D boxes / tracks → text LLM → native benchmark scores")
    parser.add_argument("--benchmark", default=default_benchmark)
    parser.add_argument("--list-benchmarks", action="store_true")
    parser.add_argument("--data", "--qa-json", dest="data", help="Native dataset path or HF dataset ID")
    parser.add_argument("--split")
    parser.add_argument("--data-format", choices=["auto", "native", "legacy-vsi"], default="auto")
    parser.add_argument("--dataset", default="all", help="VSI source filter: scannet,arkitscenes,scannetpp or all")
    parser.add_argument("--vsi-metric-protocol", choices=["physbrain", "project"], default="physbrain",
                        help="project reproduces the existing acc.py; its overall is sample-weighted")
    parser.add_argument("--geometry", "--bbox-json", dest="geometry")
    parser.add_argument("--units", choices=["m", "cm", "mm"], help="Required for legacy bbox dictionaries")
    parser.add_argument("--coordinate-frame", help="Explicit legacy coordinate frame description")
    parser.add_argument("--require-tracks", action="store_true")
    parser.add_argument("--representation", choices=["3d", "2d", "none"], default="3d")
    parser.add_argument("--run-label", default="", help="Experiment identity, e.g. full_pred3d or small_gt2d_pred3d")
    parser.add_argument("--model", default="unspecified", help="Served LLM name; the encoder checkpoint belongs in geometry provenance")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint including /v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--predictions", help="Score an existing prediction JSON/JSONL without contacting an LLM")
    parser.add_argument("--output", default="results/evaluation.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Load real data and validate every selected geometry record; no API requests")
    parser.add_argument("--export-manifest", help="Export public questions/IDs/media references; no answer labels")
    parser.add_argument("--export-media-dir", help="With --export-manifest, save embedded images/frames/video for encoder input")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16, help="Bound media decoding and pending requests")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=150000)
    parser.add_argument('--geometry-decimals', type=int, choices=range(9), default=None,
                        help='Explicit decimal precision for geometry text; saved geometry stays unchanged')
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra-body", type=json.loads, default={}, help='Server-specific JSON, e.g. {"chat_template_kwargs":{"enable_thinking":false}}')
    parser.add_argument("--save-prompts", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.list_benchmarks:
        for spec in BENCHMARKS.values():
            print(f"{spec.name:24} {spec.group:23} {spec.priority:10} {spec.dataset} [{spec.split}]")
        return
    if not args.benchmark:
        parser.error("--benchmark is required")
    if args.export_media_dir and not args.export_manifest:
        parser.error("--export-media-dir requires --export-manifest")
    if min(args.batch_size, args.concurrency, args.max_tokens, args.max_prompt_chars) <= 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("Batch/concurrency/token/time limits must be positive; retries must be nonnegative")
    if not isinstance(args.extra_body, dict):
        parser.error("--extra-body must be a JSON object")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    from .runner import run
    try:
        report = run(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report.get("status_counts", {}).get("error", 0):
            parser.exit(3, "Results saved, but inference requests failed. Inspect status_counts and use --resume after fixing the service.\n")
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        parser.exit(2, f"4DEvalKit: {exc}\n")
