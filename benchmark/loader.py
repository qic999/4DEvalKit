"""Native benchmark adapters with bounded media preparation and stable row IDs."""
from __future__ import annotations

import importlib
import inspect
import hashlib
import random
from collections import Counter
from pathlib import Path

from core.io import read_json, digest


def scoring_fingerprint(sample):
    """Include scoring-only annotations (e.g. masks) in resume identity, never prompts."""
    def encode(value):
        if isinstance(value, dict):
            return {str(k): encode(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(v) for v in value]
        if isinstance(value, bytes):
            return {"bytes_sha256": hashlib.sha256(value).hexdigest()}
        if hasattr(value, "mode") and hasattr(value, "tobytes") and hasattr(value, "size"):
            return {"image_sha256": hashlib.sha256(value.tobytes()).hexdigest(), "size": list(value.size), "mode": value.mode}
        if hasattr(value, "tolist"):
            return encode(value.tolist())
        if isinstance(value, Path):
            return str(value)
        return value
    return digest(encode({"answer": sample.get("answer"), "metadata": sample.get("metadata", {})}))


class LegacyVSI:
    """Adapt the project's existing vsibench_qa.json without loading videos."""
    def __init__(self, native):
        self.native = native

    def prepare_dataset(self, rows):
        result = []
        for row in rows:
            options = row.get("options", [])
            question = row["question"]
            if options:
                question += "\n" + "\n".join(options)
            question += "\nAnswer with only the option letter or a numeric value, as requested."
            result.append({"question": question, "answer": str(row["ground_truth"]).strip().lower(),
                "metadata": {key:row.get(key) for key in ["id", "dataset", "scene_name", "question_type", "options", "video_path"]}})
        return result

    def evaluate_results(self, samples, outputs):
        return self.native.evaluate_results(samples, outputs)

    def compute_statistics(self, results):
        return self.native.compute_statistics(results)


def make_adapter(spec, data, split, model_name):
    prefix = "benchmark" if spec.module == "dynamic" else "benchmark.physbrain"
    module = importlib.import_module(f"{prefix}.{spec.module}")
    cls = getattr(module, spec.class_name)
    available = inspect.signature(cls).parameters
    kwargs = dict(dataset_name=data, dataset_path=data, split=split, task_name=spec.name,
                  model_name=model_name, backbone="qwen3", debug=False, thinking_model=False)
    if spec.name == "RoboVQA":
        kwargs.update(expected_num_frames=16, prompt_policy="raw")
    if spec.name in {"CV-Bench", "SAT"}:
        kwargs["subset"] = "default"
    return cls(**{key:value for key,value in kwargs.items()
                  if key in available or spec.module == "dynamic"})


def load_local_generic(path, split):
    from datasets import load_dataset
    p = Path(path)
    if p.is_file():
        if p.suffix in {".json", ".jsonl"}:
            return read_json(p)
        if p.suffix == ".parquet":
            return load_dataset("parquet", data_files={split: str(p)}, split=split)
        raise ValueError(f"Unsupported local annotation format {p.suffix}")
    files = sorted(p.rglob(f"*{split}*.parquet"))
    if not files:
        # A single partition named data/ is common in evaluation-only mirrors.
        candidates = sorted(p.rglob("*.parquet"))
        if len(candidates) == 1:
            files = candidates
    if not files:
        raise FileNotFoundError(f"No unambiguous parquet split {split!r} under {p}; pass its annotation file directly")
    return load_dataset("parquet", data_files={split: [str(x) for x in files]}, split=split)


class BenchmarkSession:
    def __init__(self, spec, *, data=None, split=None, data_format="auto", model_name="model", dataset_filter=None,
                 limit=None, seed=0, batch_size=16):
        self.spec, self.data = spec, data or spec.dataset
        self.split = split or spec.split
        self.seed, self.batch_size = seed, batch_size
        self.adapter = make_adapter(spec, self.data, self.split, model_name)
        if spec.name == "BLINK":
            # PhysBrain Table 4 evaluates only these three visual-spatial tasks.
            self.adapter.SUBSETS = ["Counting", "Relative_Depth", "Spatial_Relation"]
        p = Path(self.data)
        self.legacy = data_format == "legacy-vsi"
        if spec.name == "VSI-Bench" and p.is_file() and p.suffix in {".json", ".jsonl"}:
            self.legacy = data_format in {"auto", "legacy-vsi"}
        if self.legacy:
            if spec.name != "VSI-Bench":
                raise ValueError("legacy-vsi input is only supported for VSI-Bench")
            self.raw = read_json(p)
            self.adapter = LegacyVSI(self.adapter)
        elif spec.module == "dynamic" or spec.module in {"threedsrbench", "mindcube", "viewspatial", "mmsi_bench", "erqa_plus", "roborefit", "robovqa", "vlabench"}:
            self.raw = self.adapter.load_dataset()
        elif p.exists() and spec.module not in {"blink", "robospatial", "refspatial"}:
            self.raw = load_local_generic(p, self.split)
        else:
            self.raw = self.adapter.load_dataset()
        if not hasattr(self.raw, "__len__") or len(self.raw) == 0:
            raise ValueError("No benchmark samples loaded")
        self.indices = list(range(len(self.raw)))
        if dataset_filter and dataset_filter != "all":
            allowed = set(dataset_filter.split(","))
            if spec.name != "VSI-Bench":
                raise ValueError("--dataset filters VSI-Bench scene sources only")
            self.indices = [i for i in self.indices if self.raw[i].get("dataset") in allowed]
        self.available_count = len(self.indices)
        if limit is not None:
            if limit <= 0:
                raise ValueError("--limit must be positive")
            self.indices = self.indices[:limit]
            if spec.name == "3DSRBench":
                # Do not turn circular evaluation into easier single-variant QA.
                base = self.adapter._base_qid
                groups = {base(self.raw[i]["qid"]) for i in self.indices}
                self.indices = [i for i in range(len(self.raw)) if base(self.raw[i]["qid"]) in groups]
            if spec.name == "DSI-Bench" and self.split == "all":
                groups = {self.raw[i]["_base_index"] for i in self.indices}
                self.indices = [i for i in range(len(self.raw)) if self.raw[i]["_base_index"] in groups]
        if not self.indices:
            raise ValueError("Dataset selection is empty")
        self.selected_count = len(self.indices)
        self.protocol = ("vlm4d_direct_choice_v1" if spec.name == "VLM4D" else
                         "dynamic_direct_choice_v1" if spec.module == "dynamic" else
                         "physbrain_" + "4b37ca2")

    def batches(self):
        random.seed(self.seed)
        seen = set()
        for start in range(0, len(self.indices), self.batch_size):
            indices = self.indices[start:start+self.batch_size]
            rows = self.raw.select(indices) if hasattr(self.raw, "select") else [self.raw[i] for i in indices]
            samples = self.adapter.prepare_dataset(rows)
            if len(samples) != len(indices):
                raise ValueError("Adapter changed sample cardinality; refusing to silently skip benchmark rows")
            batch = []
            for global_index, sample in zip(indices, samples):
                meta = sample.setdefault("metadata", {})
                meta["idx"] = global_index
                sample_id = f"{self.spec.slug}:{global_index}"
                source_id = next((meta[key] for key in ["id", "question_id", "index", "qid"]
                                  if meta.get(key) is not None), None)
                if self.spec.name == "3DSRBench":
                    source_id = meta.get("index")  # Keep FlipEval/circular variants distinct.
                aliases = []
                if source_id is not None:
                    aliases.append(str(source_id))
                if self.spec.name == "VSI-Bench" and meta.get("scene_name"):
                    aliases.append(meta["scene_name"])
                question = sample["question"]
                # Native VSI stores choices in metadata rather than the question.
                if self.spec.name == "VSI-Bench" and not self.legacy and meta.get("options"):
                    question += "\n" + "\n".join(meta["options"])
                if sample_id in seen:
                    raise ValueError(f"Duplicate sample ID {sample_id}")
                seen.add(sample_id)
                task = meta.get("question_type", meta.get("task", meta.get("category", "unknown")))
                batch.append({"sample_id": sample_id, "source_id": source_id, "aliases": aliases,
                              "question": question, "task": str(task), "sample": sample,
                              "answer_fingerprint": scoring_fingerprint(sample), "row_index": global_index})
            yield batch


def media_manifest(example, export_media_dir=None):
    sample = example["sample"]
    meta = sample["metadata"]
    public = {key: meta[key] for key in ["scene_name", "video_path", "image_path", "image_paths", "num_images",
             "time_start", "time_end", "augmentation"] if meta.get(key) is not None}
    def describe(value):
        if isinstance(value, (str, Path)):
            return {"path": str(value)}
        if isinstance(value, (list, tuple)):
            return [describe(x) for x in value]
        if hasattr(value, "size") and hasattr(value, "mode"):
            return {"embedded_image": True, "size": list(value.size)}
        if isinstance(value, dict):
            return {"embedded_or_lazy_media": True}
        return {"embedded_media": value is not None}
    media = {key: describe(sample[key]) for key in ["image", "video"] if key in sample}
    if export_media_dir:
        from core.media import export_media
        directory = Path(export_media_dir) / example["sample_id"].replace(":", "_")
        media = {key: export_media(sample[key], directory, stem=key, kind=key) for key in media}
    return {"sample_id": example["sample_id"], "source_id": example["source_id"],
            "geometry_aliases": example["aliases"], "question": example["question"], "task": example["task"],
            "media": media,
            "input_metadata": public}
