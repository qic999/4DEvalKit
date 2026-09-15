"""Bounded inference, durable resume and protocol-preserving score aggregation."""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark.loader import BenchmarkSession, media_manifest
from scripts.benchmark_registry import UPSTREAM_REVISION, get_spec
from .answers import final_text
from .geometry import GeometryStore
from .inference import APIInferenceEngine, ReplayEngine
from .io import digest, dumps, read_json, source_signature, write_json
from .prompts import PROMPT_VERSION, make_messages

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def output_lock(path):
    lock_path = Path(str(path) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Another process is writing {path}")
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_journal(path):
    if not path.exists():
        return {}
    content = path.read_bytes()
    lines = content.splitlines(keepends=True)
    rows, valid_end = {}, 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if index == len(lines)-1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(valid_end)
                break
            raise ValueError(f"Corrupt prediction journal at line {index+1}")
        rows[record["sample_id"]] = record
        valid_end += len(line)
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as stream:
            stream.write(b"\n")
    return rows


def primary_metric(spec, statistics):
    if spec.group == "grounding":
        key = "non_strict_overall_score" if spec.name == "RoboSpatial-Home" else "non_strict_micro_f1"
    elif spec.name == "Q-Spatial-Bench":
        key = "success_rate"
    else:
        key = spec.score_key
    value = statistics.get(key)
    if value is None:
        return {"field": key, "value": None, "score_100": None}
    score = float(value) * spec.score_scale
    if spec.group == "trajectory_generation":
        # Vendored field is normalized ERROR, despite its misleading name.
        return {"field": key, "value": float(value), "score_100": 100.0 - score,
                "transform": "100 - normalized RMSE error; valid trajectories only",
                "valid_samples": statistics.get("valid_samples"), "total_samples": statistics.get("total_samples")}
    return {"field": key, "value": float(value), "score_100": score}


def run(args):
    spec = get_spec(args.benchmark)
    output = Path(args.output)
    if output.suffix != ".json":
        raise ValueError("--output must end in .json; a sibling .jsonl journal is written during inference")
    geometry = GeometryStore(args.geometry, units=args.units, coordinate_frame=args.coordinate_frame) if args.geometry else None
    replay = ReplayEngine(args.predictions) if args.predictions else None
    if not geometry and not replay and not args.export_manifest:
        raise ValueError("Provide --geometry (model predictions) or --predictions (offline scoring)")
    if args.require_tracks and geometry is None:
        raise ValueError("--require-tracks needs --geometry")
    session = BenchmarkSession(spec, data=args.data, split=args.split, data_format=args.data_format,
        model_name=args.model, dataset_filter=args.dataset, limit=args.limit, seed=args.seed, batch_size=args.batch_size)
    if args.vsi_metric_protocol == "project":
        if spec.name != "VSI-Bench":
            raise ValueError("--vsi-metric-protocol project applies only to VSI-Bench")
        from benchmark.project_vsi import ProjectVSI
        session.adapter = ProjectVSI(session.adapter)
        session.protocol = "project_vsi_acc_v1"
    config = {"benchmark": spec.name, "source": source_signature(session.data), "split": session.split,
              "data_format": args.data_format, "dataset_filter": args.dataset, "limit": args.limit,
              "selected_count": session.selected_count, "model": args.model, "run_label": args.run_label, "base_url": args.base_url,
              "max_tokens": args.max_tokens, "temperature": args.temperature, "seed": args.seed,
              "extra_body": args.extra_body, "prompt_version": PROMPT_VERSION,
              "metric_protocol": session.protocol, "upstream_revision": UPSTREAM_REVISION,
              "representation": args.representation, "require_tracks": args.require_tracks,
              "units": args.units, "coordinate_frame": args.coordinate_frame,
              "geometry_digest": digest(geometry.scenes) if geometry else None,
              "replay_digest": digest(replay.outputs) if replay else None,
              "python": platform.python_version()}
    if getattr(args, 'geometry_decimals', None) is not None:
        config['geometry_decimals'] = args.geometry_decimals
    if args.export_manifest:
        if Path(args.export_manifest).exists():
            raise FileExistsError("Manifest output exists; choose a new path")
        manifest = [media_manifest(example, args.export_media_dir) for batch in session.batches() for example in batch]
        write_json(args.export_manifest, {"benchmark": spec.name, "source": config["source"], "split": session.split,
                                         "samples": manifest})
        return {"exported_samples": len(manifest), "manifest": str(args.export_manifest)}
    if args.dry_run:
        if output.exists():
            raise FileExistsError(f"Dry-run output exists: {output}; choose a new filename")
        statuses = Counter()
        preview = None
        for batch in session.batches():
            for example in batch:
                if geometry:
                    key, scene = geometry.lookup(example["sample_id"], example["aliases"])
                    if args.require_tracks and not scene.get("tracks"):
                        raise ValueError(f"{example['sample_id']} has no tracks")
                    messages = make_messages(example["question"], scene, args.representation, args.max_prompt_chars,
                                             decimals=getattr(args,'geometry_decimals',None))
                    preview = preview or {"sample_id": example["sample_id"], "geometry_key": key, "messages": messages}
                    statuses["geometry_matched"] += 1
                if replay:
                    statuses[replay.infer(example["sample_id"], example["source_id"],
                        question=example["question"], metadata=example["sample"]["metadata"])["status"]] += 1
        report = {"config": config, "selected_count": session.selected_count, "checks": dict(statuses), "preview": preview}
        write_json(output, report)
        return {"dry_run": True, "selected_count": session.selected_count, "checks": dict(statuses), "output": str(output)}
    engine = None
    if not replay:
        if not args.base_url:
            raise ValueError("Live inference requires --base-url pointing to a deployed LLM's /v1 endpoint")
        if args.model == "unspecified":
            raise ValueError("Live inference requires --model (the served reasoning LLM name)")
        engine = APIInferenceEngine(model=args.model, base_url=args.base_url, api_key_env=args.api_key_env,
            timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens,
            temperature=args.temperature, seed=args.seed, extra_body=args.extra_body)
    manifest_path = output.with_suffix(".manifest.json")
    journal_path = output.with_suffix(".jsonl")
    with output_lock(output):
        if manifest_path.exists():
            if not args.resume:
                raise FileExistsError(f"Run exists at {output}; use --resume or a new output filename")
            previous_config = read_json(manifest_path)
            if previous_config != config:
                changed = sorted(key for key in set(previous_config) | set(config) if previous_config.get(key) != config.get(key))
                raise ValueError(f"Resume configuration mismatch: {changed}")
        elif output.exists() or journal_path.exists():
            raise FileExistsError("Output exists without its run manifest; choose a new output filename")
        else:
            write_json(manifest_path, config)
        saved = read_journal(journal_path) if args.resume else {}
        all_results, statuses, reused = [], Counter(), 0
        with journal_path.open("a", encoding="utf-8") as journal, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for batch in session.batches():
                ready, futures = {}, {}
                for example in batch:
                    sample_id = example["sample_id"]
                    key, scene, messages = None, None, None
                    if geometry:
                        key, scene = geometry.lookup(sample_id, example["aliases"])
                        if args.require_tracks and not scene.get("tracks"):
                            raise ValueError(f"{sample_id} has no tracks")
                        messages = make_messages(example["question"], scene, args.representation, args.max_prompt_chars,
                                                 decimals=getattr(args,'geometry_decimals',None))
                    fingerprint = digest({"messages": messages, "question": example["question"],
                                          "answer_fingerprint": example["answer_fingerprint"]})
                    context = {"sample_id": sample_id, "source_id": example["source_id"], "task": example["task"],
                               "sample_fingerprint": fingerprint, "geometry_key": key,
                               "provenance": scene.get("provenance", {}) if scene else {}}
                    if args.save_prompts:
                        context["messages"] = messages
                    previous = saved.get(sample_id)
                    if previous and previous["sample_fingerprint"] != fingerprint:
                        raise ValueError(f"Resume sample changed: {sample_id}")
                    if previous and previous.get("status") == "ok":
                        ready[sample_id] = previous
                        reused += 1
                    elif replay:
                        ready[sample_id] = {**context, **replay.infer(sample_id, example["source_id"],
                            question=example["question"], metadata=example["sample"]["metadata"])}
                        journal.write(dumps(ready[sample_id]) + "\n"); journal.flush()
                    else:
                        futures[pool.submit(engine.infer, messages)] = context
                for future in as_completed(futures):
                    context = futures[future]
                    record = {**context, **future.result()}
                    ready[context["sample_id"]] = record
                    journal.write(dumps(record) + "\n"); journal.flush()
                records = [ready[e["sample_id"]] for e in batch]
                extract = (lambda text: text) if session.protocol == "project_vsi_acc_v1" else final_text
                outputs = [extract(row["raw_output"]) if row.get("status") == "ok" else "" for row in records]
                scored = session.adapter.evaluate_results([e["sample"] for e in batch], outputs)
                if len(scored) != len(batch):
                    raise RuntimeError("Scorer dropped samples")
                for score, record in zip(scored, records):
                    all_results.append({**score, **record})
                    statuses[record["status"]] += 1
                logger.info("%s: %d/%d rows scored (%d reused)", spec.name, len(all_results), session.selected_count, reused)
        statistics = session.adapter.compute_statistics(all_results)
        # Upstream VSI includes a duplicate copy of all results in statistics.
        statistics.pop("predictions", None)
        result = {"benchmark": spec.name, "model": args.model, "group": spec.group,
                  "config": config, "statistics": statistics, "primary_metric": primary_metric(spec, statistics),
                  "num_samples": len(all_results), "available_count": session.available_count,
                  "selection_is_partial": len(all_results) < session.available_count,
                  "status_counts": dict(statuses), "reused_predictions": reused, "results": all_results}
        write_json(output, result)
    return {key: result[key] for key in ["benchmark", "model", "primary_metric", "num_samples", "status_counts", "reused_predictions"]}
