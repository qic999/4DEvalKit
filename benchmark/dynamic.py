"""Native STI/VLM4D/DSI annotations with explicit, deterministic MCQ scoring.

VLM4D's published free-text results use model judges and manual verification.
Our direct-choice protocol is intentionally named differently in result metadata.
"""
from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from core.answers import choice_letter, final_text
from core.io import read_json

DSI_CATEGORIES = ["object_motion_static_camera", "object_motion_moving_camera", "camera_motion_static_scene",
                  "camera_motion_dynamic_scene", "object_camera_distance", "object_camera_orientation"]


def choices_dict(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # DSI-Bench's released CSV uses "A: ...; B: ...; C: ...".
                matches = list(re.finditer(r"(?:^|;\s*)([A-Z]):\s*", value))
                if not matches:
                    raise ValueError("Unrecognized multiple-choice option format")
                value = {m.group(1): value[m.end():matches[i+1].start() if i+1 < len(matches) else len(value)].strip()
                         for i, m in enumerate(matches)}
    if isinstance(value, list):
        return {chr(65+i): str(v) for i,v in enumerate(value)}
    if isinstance(value, dict) and value:
        return {str(k).strip().upper(): str(v) for k,v in value.items()}
    raise ValueError("A multiple-choice sample needs nonempty choices")


def load_rows(path):
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if path.suffix in {".csv", ".tsv"}:
        with path.open() as stream:
            return list(csv.DictReader(stream, delimiter="\t" if path.suffix == ".tsv" else ","))
    rows = read_json(path)
    if isinstance(rows, dict):
        rows = rows.get("data", rows.get("questions", rows.get("samples")))
    if not isinstance(rows, list):
        raise ValueError(f"Expected annotation records in {path}")
    return rows


class DynamicDataset:
    def __init__(self, task_name, dataset_name, split=None, **kwargs):
        self.task_name, self.dataset_name, self.split = task_name, dataset_name, split

    def load_dataset(self):
        path = Path(self.dataset_name)
        if not path.exists():
            from huggingface_hub import snapshot_download
            path = Path(snapshot_download(self.dataset_name, repo_type="dataset",
                        allow_patterns=["*.parquet", "*.json", "*.csv", "**/*.parquet", "**/*.json", "**/*.csv"]))
        if path.is_file():
            rows = load_rows(path)
            if self.task_name == "DSI-Bench":
                if self.split == "all":
                    raise ValueError("DSI --split all needs a directory containing all four variant CSV files")
                rows = [{**row, "_augmentation": self.split or "std", "_base_index": i} for i,row in enumerate(rows)]
            return rows
        if self.task_name == "STI-Bench":
            files = list(path.rglob("qa.parquet"))
        elif self.task_name == "VLM4D":
            target = (self.split or "real_mc") + ".json"
            files = list(path.rglob(target))
        else:
            variant = self.split or "std"
            variants = ["std", "reverse", "hflip", "reverse_hflip"] if variant == "all" else [variant]
            rows = []
            for aug in variants:
                candidates = list(path.rglob(aug + ".csv"))
                if len(candidates) != 1:
                    raise FileNotFoundError(f"Expected one {aug}.csv under {path}; use an explicit annotation file")
                for i,row in enumerate(load_rows(candidates[0])):
                    rows.append({**row, "_augmentation": aug, "_base_index": i})
            return rows
        if len(files) != 1:
            raise FileNotFoundError(f"Expected one annotation file for {self.task_name}/{self.split} under {path}; found {len(files)}")
        return load_rows(files[0])

    def prepare_dataset(self, rows):
        prepared = []
        for index, row in enumerate(rows):
            question = row.get("Question", row.get("question"))
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Missing question in dynamic row {index}")
            choices = row.get("Candidates", row.get("choices", row.get("options")))
            if choices is None:
                choices = {key: row[key] for key in "ABCDE" if row.get(key) is not None}
            choices = choices_dict(choices)
            answer = row.get("Answer", row.get("answer", row.get("GT")))
            label = choice_letter(answer, choices)
            if label is None:
                # VLM4D real validation_633 has identical correct C and D text.
                # Preserve that annotation and accept both semantically identical
                # choices instead of silently dropping the row or choosing C.
                label = [key for key,value in choices.items() if str(answer).strip().casefold() == value.strip().casefold()]
                if not label:
                    raise ValueError(f"Ground truth does not match any choice in row {index}")
            video = row.get("Video", row.get("video", row.get("video_path", row.get("video_name", row.get("relative_path")))))
            sample_id = row.get("id", row.get("ID", index))
            meta = {"id": sample_id, "video_path": video,
                    "question_type": row.get("Task", row.get("cate", row.get("question_type", "unknown"))),
                    "scene": row.get("scene", row.get("Source", "unknown")), "choices": choices}
            if self.task_name == "DSI-Bench" and "cate" in row:
                category = int(float(row["cate"]))
                if not 0 <= category < len(DSI_CATEGORIES):
                    raise ValueError(f"Unknown DSI category {category}")
                meta.update(category_id=category, question_type=DSI_CATEGORIES[category])
            if self.task_name == "VLM4D":
                # The authors' acc_final_statistics.py uses this native ID
                # partition; synthetic FP strata are scoring-only annotations.
                native_id = re.fullmatch(r"validation_(\d+)", str(sample_id))
                if self.split == "real_mc" and native_id:
                    meta["scoring_stratum"] = "real_exocentric" if int(native_id[1]) <= 922 else "real_egocentric"
                elif self.split == "synthetic_mc":
                    text_answer = str(answer).strip().lower()
                    meta["scoring_stratum"] = ("synthetic_false_positive" if text_answer == "no" or text_answer.startswith("no ")
                                               else "synthetic_directional")
            if self.task_name == "STI-Bench":
                meta.update(time_start=row.get("time_start"), time_end=row.get("time_end"))
                prefix = str(row.get("Prompt") or "").strip()
                if prefix:
                    question = prefix + "\n" + question
            if "_augmentation" in row:
                meta.update(augmentation=row["_augmentation"], base_index=row["_base_index"])
                meta["id"] = f"{row['_augmentation']}:{row['_base_index']}"
            prepared.append({"question": question.strip() + "\n" + "\n".join(
                f"{key}. {value}" for key,value in choices.items()) + "\nAnswer with only the option letter.",
                "answer": label, "video": video, "metadata": meta})
        return prepared

    def evaluate_results(self, samples, outputs):
        results = []
        for sample, output in zip(samples, outputs):
            parsed = choice_letter(output, sample["metadata"]["choices"])
            accepted = sample["answer"] if isinstance(sample["answer"], list) else [sample["answer"]]
            exact_text_hits = [key for key,value in sample["metadata"]["choices"].items()
                               if value.strip().casefold() == final_text(output).casefold()]
            results.append({"metadata": sample["metadata"], "ground_truth": sample["answer"],
                            "processed_answer": parsed, "is_correct": parsed in accepted or
                            bool(exact_text_hits) and all(key in accepted for key in exact_text_hits)})
        return results

    def compute_statistics(self, results):
        by_type, by_scene, groups, by_stratum = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        for row in results:
            meta = row["metadata"]
            by_type[str(meta["question_type"])].append(int(row["is_correct"]))
            by_scene[str(meta.get("scene", "unknown"))].append(int(row["is_correct"]))
            if "scoring_stratum" in meta:
                by_stratum[meta["scoring_stratum"]].append(int(row["is_correct"]))
            if "base_index" in meta:
                groups[meta["base_index"]].append(row)
        avg = lambda rows: sum(rows) / len(rows) if rows else 0.0
        stats = {"overall_accuracy": avg([int(r["is_correct"]) for r in results]),
                 "total_samples": len(results), "per_type": {k: {"accuracy": avg(v), "count": len(v)} for k,v in by_type.items()},
                 "per_scene": {k: {"accuracy": avg(v), "count": len(v)} for k,v in by_scene.items()}}
        if by_stratum:
            stats["per_stratum"] = {k: {"accuracy": avg(v), "count": len(v)} for k,v in by_stratum.items()}
        if groups:
            expected = {"std", "reverse", "hflip", "reverse_hflip"}
            if all({r["metadata"]["augmentation"] for r in g} == expected and len(g) == 4 for g in groups.values()):
                stats["robust_accuracy"] = {str(n): avg([int(sum(r["is_correct"] for r in g) >= n)
                    for g in groups.values()]) for n in range(1,5)}
            else:
                stats["robust_accuracy"] = None
                stats["robust_note"] = "Requires all four augmentations for every base question."
        return stats
