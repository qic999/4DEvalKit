"""Read-only checks against locally available benchmark data and existing VSI predictions.

Config keys: datasets [{benchmark,data,split?,limit?}], vsi_qa?, vsi_predictions?,
vsi_geometry? (legacy 9D world_z_up meters). Outputs never claim new model inference.
"""
import argparse
from pathlib import Path
import subprocess
import sys

from benchmark.loader import BenchmarkSession, media_manifest
from core.io import read_json, write_json
from scripts.benchmark_registry import get_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"kind": "implementation_validation_not_new_model_evaluation", "adapters": [], "commands": []}
    for task in config.get("datasets", []):
        print(f"Checking native annotations: {task['benchmark']}", flush=True)
        session = BenchmarkSession(get_spec(task["benchmark"]), data=task["data"], split=task.get("split"),
                                   limit=task.get("limit", 2), batch_size=16)
        examples = [sample for batch in session.batches() for sample in batch]
        manifests = [media_manifest(sample) for sample in examples]
        write_json(out / (session.spec.slug + "_" + session.split.replace("/", "_") + ".manifest.json"), {"samples": manifests})
        report["adapters"].append({"benchmark": task["benchmark"], "split": session.split,
                                   "available": session.available_count, "prepared": len(examples), "status": "passed"})
    if config.get("vsi_qa"):
        common = [sys.executable, "eval_vsi_bench.py", "--data", config["vsi_qa"], "--dataset", "scannet",
                  "--model", "existing_qwen35_9b_predictions", "--run-label", "existing_predictions_replay"]
        commands = []
        if config.get("vsi_geometry"):
            commands.append(common + ["--geometry", config["vsi_geometry"], "--units", "m", "--coordinate-frame", "world_z_up",
                            "--dry-run", "--output", str(out / "vsi_geometry.dryrun.json")])
        if config.get("vsi_predictions"):
            for protocol in ["physbrain", "project"]:
                commands.append(common + ["--predictions", config["vsi_predictions"], "--vsi-metric-protocol", protocol,
                                          "--output", str(out / f"vsi_replay_{protocol}.json")])
        for command in commands:
            log = out / (Path(command[-1]).stem + ".log")
            with log.open("w") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            report["commands"].append({"command": command, "returncode": result.returncode, "log": str(log)})
            print(f"{command[-1]}: exit {result.returncode}", flush=True)
        if config.get("vsi_predictions") and all(c["returncode"] == 0 for c in report["commands"]):
            from benchmark.legacy.vsi_acc import process_vsibench_single, result_correct, result_total
            buckets = process_vsibench_single(config["vsi_predictions"], "scannet")
            expected = result_correct(buckets) / result_total(buckets)
            replay = read_json(out / "vsi_replay_project.json")
            actual = replay["statistics"]["overall_score"]
            if abs(actual - expected) > 1e-12 or replay["num_samples"] != result_total(buckets):
                raise ValueError(f"Legacy VSI parity failed: {expected} versus {actual}")
            report["legacy_vsi_parity"] = {"status": "passed", "samples": result_total(buckets), "score_100": actual*100}
    report["passed"] = all(c["returncode"] == 0 for c in report["commands"])
    write_json(out / "verification.json", report)
    if not report["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
