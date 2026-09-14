"""Run a shared benchmark/LLM protocol across Full, Small and ablation inputs.

Paths in a suite config are relative to the repository root (not the config file).
Each job gets its own result, resume journal, and log. A failed job does not erase
finished results; --resume reruns failed/missing samples and verifies fingerprints.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from core.io import read_json, write_json
from scripts.benchmark_registry import get_spec

ROOT = Path(__file__).resolve().parents[1]


def command_args(options):
    result = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            result.append(flag)
        elif value is not False and value is not None:
            result += [flag, json.dumps(value) if isinstance(value, (dict, list)) else str(value)]
    return result


def build_jobs(config, output_dir, mode, resume=False, limit=None):
    defaults = config.get("defaults", {})
    jobs = []
    if not config.get("benchmarks") or not config.get("variants"):
        raise ValueError("Suite requires nonempty benchmarks and variants")
    for variant in config["variants"]:
        label = variant["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError("Variant names must contain only letters, numbers, underscores and hyphens")
        for task in config["benchmarks"]:
            spec = get_spec(task["benchmark"])
            key = f"{label}/{spec.slug}"
            if any(j["job"] == key for j in jobs):
                raise ValueError(f"Duplicate suite job {key}; use separate configs for distinct splits")
            options = {**defaults, **task, **variant.get("options", {})}
            # Variant differences are geometry/representation, while defaults fix
            # the LLM and its generation settings for paired comparisons.
            geometry = variant.get("geometry", {}).get(spec.name)
            if geometry is None and variant.get("geometry_template"):
                geometry = variant["geometry_template"].format(benchmark=spec.slug)
            if geometry:
                options["geometry"] = os.path.expandvars(geometry)
            if not options.get("geometry") and not options.get("predictions"):
                raise ValueError(f"{key}: no geometry or offline predictions configured")
            options["run_label"] = label
            options["output"] = str(Path(output_dir) / label / (spec.slug + (".dryrun.json" if mode == "check" else ".json")))
            options["dry_run"] = mode == "check"
            options["resume"] = resume and mode == "run"
            if limit is not None:
                options["limit"] = limit
            jobs.append({"job": key, "benchmark": spec.name, "variant": label,
                         "output": options["output"], "options": options,
                         "command": [sys.executable, str(ROOT / "eval.py"), *command_args(options)]})
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="results/suite")
    parser.add_argument("--mode", choices=["plan", "check", "run"], default="plan")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = read_json(args.config)
    jobs = build_jobs(config, args.output_dir, args.mode, args.resume, args.limit)
    if args.mode == "plan":
        import shlex
        for job in jobs:
            print(shlex.join(job["command"]))
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statuses = []
    for job in jobs:
        print(f"Starting {job['job']}", flush=True)
        log = Path(job["output"]).with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            completed = subprocess.run(job["command"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        statuses.append({key:job[key] for key in ["job", "benchmark", "variant", "output"]} |
                        {"returncode": completed.returncode, "log": str(log)})
        write_json(output_dir / "suite_index.json", {"mode": args.mode, "jobs": statuses})
        print(f"Finished {job['job']}: exit {completed.returncode}; {log}", flush=True)
    if any(job["returncode"] != 0 for job in statuses):
        sys.exit(1)


if __name__ == "__main__":
    main()
