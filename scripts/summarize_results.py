"""Export scores and protocol/coverage metadata; never mix unlike metrics into an overall average."""
import argparse
import csv
from pathlib import Path

from core.io import read_json


def rows_from_paths(paths):
    for path in paths:
        result = read_json(path)
        if not isinstance(result, dict) or "primary_metric" not in result:
            continue
        config = result["config"]
        metric = result["primary_metric"]
        provenance = {str(r.get("provenance", {}).get("geometry_source", "unspecified")) for r in result["results"]}
        yield {"benchmark": result["benchmark"], "run_label": config.get("run_label", ""),
               "reasoning_model": result["model"], "group": result["group"],
               "score_100": metric["score_100"], "metric": metric["field"],
               "metric_protocol": config["metric_protocol"], "split": config["split"],
               "dataset_filter": config.get("dataset_filter", "all"),
               "data_source": config["source"].get("path", config["source"].get("hub_id", "")),
               "representation": config["representation"], "geometry_source": ";".join(sorted(provenance)),
               "samples": result["num_samples"], "available_samples": result["available_count"],
               "partial_selection": result["selection_is_partial"],
               "ok_completions": result["status_counts"].get("ok", 0),
               "valid_trajectories": metric.get("valid_samples", ""),
               "result_path": str(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Result JSON files or directories searched recursively")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = set()
    for name in args.inputs:
        path = Path(name)
        paths.update(path.rglob("*.json") if path.is_dir() else [path])
    rows = list(rows_from_paths(sorted(paths)))
    if not rows:
        parser.error("No scored result bundles found")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"Wrote {len(rows)} score rows to {out}")


if __name__ == "__main__":
    main()
