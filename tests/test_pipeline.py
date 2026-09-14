import base64
import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from benchmark.dynamic import DynamicDataset, choices_dict
from benchmark.loader import BenchmarkSession, make_adapter, media_manifest
from core.answers import choice_letter, final_text
from core.geometry import GeometryStore, normalize_scene
from core.io import read_json, write_json
from core.prompts import make_messages
from core.runner import run, read_journal, primary_metric
from scripts.benchmark_registry import BENCHMARKS, get_spec
from scripts.convert_geometry import raw_scene, merged_scene


def box(**extra):
    return {"instance_id": "chair_1", "category": "chair", "center": [1, 2, 3],
            "size": [1, 1, 2], "quaternion_xyzw": [0, 0, 0, 1], **extra}


def scene(**extra):
    return {"units": "m", "coordinate_frame": "world_z_up", "objects": [box()], **extra}


def settings(tmp_path, **overrides):
    data = tmp_path / "qa.json"
    if not data.exists():
        write_json(data, [{"id": "q1", "question": "Which object moved?", "choices": {"A": "chair", "B": "table"}, "answer": "chair"},
                          {"id": "q2", "question": "Which object stayed still?", "choices": {"A": "chair", "B": "table"}, "answer": "table"}])
    geometry = tmp_path / "geometry.json"
    if not geometry.exists():
        write_json(geometry, {"q1": scene(), "q2": scene()})
    options = dict(benchmark="VLM4D", data=str(data), split="real_mc", data_format="auto", dataset="all",
        vsi_metric_protocol="physbrain", geometry=str(geometry), units=None, coordinate_frame=None,
        require_tracks=False, representation="3d", run_label="test", model="test-llm", base_url=None,
        api_key_env="FOURDEVAL_TEST_API_KEY", predictions=None, output=str(tmp_path / "result.json"), resume=False,
        dry_run=False, export_manifest=None, export_media_dir=None, limit=None, batch_size=1, concurrency=2, timeout=3,
        retries=0, max_tokens=128, max_prompt_chars=20000, temperature=0.0, seed=0,
        extra_body={}, save_prompts=False, verbose=False)
    return SimpleNamespace(**(options | overrides))


@pytest.mark.parametrize("name", list(BENCHMARKS))
def test_all_adapters_import_without_gpu(name):
    spec = get_spec(name)
    adapter = make_adapter(spec, spec.dataset, spec.split, "test")
    assert callable(adapter.prepare_dataset) and callable(adapter.evaluate_results)


def test_legacy_requires_declared_units_and_frame(tmp_path):
    path = tmp_path / "legacy.jsonl"
    write_json(path, {"scene": [{"label": "chair", "bbox_3d": [1, 2, 3, 1, 1, 1, .2, .3, .4]}]})
    with pytest.raises(ValueError, match="units"):
        GeometryStore(path)
    obj = GeometryStore(path, units="m", coordinate_frame="world_z_up").scenes["scene"]["objects"][0]
    assert abs(obj["quaternion_xyzw"][0]) > .01  # Roll was not discarded.


@pytest.mark.parametrize("change", [{"size": [1, 0, 2]}, {"center": [float("nan"), 0, 0]},
                                   {"quaternion_xyzw": [0, 0, 0, 0]}])
def test_invalid_geometry_fails(change):
    with pytest.raises(ValueError):
        normalize_scene(scene(objects=[box(**change)]))


def test_answer_annotations_rejected_and_provenance_hidden():
    with pytest.raises(ValueError, match="Answer-only"):
        normalize_scene(scene(layout={"answer": "secret"}))
    data = normalize_scene(scene(provenance={"geometry_source": "oracle", "checkpoint": "secret-checkpoint"}))
    messages = make_messages("Where is the chair?", data)
    assert "secret-checkpoint" not in str(messages)
    assert all(isinstance(m["content"], str) for m in messages)


def test_ablations_remove_3d_and_reject_missing_2d():
    data = normalize_scene(scene(objects=[box(bbox_2d=[1, 2, 30, 40])]))
    prompt = make_messages("Which is nearer?", data, "2d")[-1]["content"]
    assert "bbox_2d" in prompt and '"center"' not in prompt and '"quaternion_xyzw"' not in prompt
    assert "chair_1" not in make_messages("Which?", data, "none")[-1]["content"]
    with pytest.raises(ValueError, match="bbox_2d"):
        make_messages("Which?", normalize_scene(scene()), "2d")


def test_dynamic_times_must_increase():
    observations = [{k:v for k,v in box(timestamp=t).items() if k not in {"category", "instance_id"}} for t in [1, 1]]
    with pytest.raises(ValueError, match="strictly increasing"):
        normalize_scene(scene(tracks=[{"track_id": "c", "observations": observations}]))


def test_converter_preserves_full_quaternion():
    out = merged_scene([{"label": "c", "bbox_3d": [1, 2, 3, 1, 2, 3, .5, .5, .5, .5]}],
        units="m", coordinate_frame="world_z_up", quaternion_order="xyzw", provenance={})
    assert out["objects"][0]["quaternion_xyzw"] == [.5, .5, .5, .5]


def test_raw_converter_motion_scale_pose_and_dedup(tmp_path):
    identity = [[1,0,0,10], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    data = {"slot_id": "1", "label": "chair", "bbox_3d": [1,0,0,1,2,3],
            "quaternion": [1,0,0,0], "conf3d": 2, "camera_pose": identity,
            "intrinsics": [[100,0,50,0],[0,100,50,0],[0,0,1,0],[0,0,0,1]]}
    for clip in [0, 1]:
        for frame in [0, 1]:
            row = dict(data, frame_path=f"f{frame}", bbox_3d=[1+frame,0,0,1,2,3])
            write_json(tmp_path / "json" / str(clip) / f"f{frame}.jsonl", row)
    kwargs = dict(timestamps={"f0": 0.0, "f1": 0.5}, poses=None, record_gt_poses=True,
                  ids_global=True, track_map=None, metric_scale=2.5, coordinate_frame="world_z_up", provenance={})
    out = raw_scene(tmp_path, **kwargs)
    observations = out["tracks"][0]["observations"]
    assert len(observations) == 2
    assert observations[0]["center"] == [12.5, 0, 0]
    assert observations[1]["center"] == [15, 0, 0]
    assert observations[0]["size"] == [2.5, 5, 7.5]
    assert out["provenance"]["overlapping_observations_deduplicated"] == 2
    assert out["cameras"][0]["intrinsics"] == [[100,0,50],[0,100,50],[0,0,1]]
    with pytest.raises(ValueError, match="clip-local"):
        raw_scene(tmp_path, **(kwargs | {"ids_global": False}))
    with pytest.raises(ValueError, match="Choose exactly"):
        raw_scene(tmp_path, **(kwargs | {"record_gt_poses": False}))


def test_dynamic_native_options_and_strict_answer():
    assert choices_dict("A: Up; B: Left; C: Right") == {"A": "Up", "B": "Left", "C": "Right"}
    assert choice_letter("There is A chair, but uncertain", {"A": "chair", "B": "table"}) is None
    assert final_text("<think>Answer A") == ""
    assert final_text("<think>A</think><answer>B</answer>") == "B"
    assert choice_letter(0, {"A": "1", "B": "0"}) == "B"


def test_dsi_robust_accuracy_needs_all_variants():
    adapter = DynamicDataset("DSI-Bench", "unused")
    results = [{"is_correct": i < 3, "metadata": {"question_type": "motion", "base_index": 0, "augmentation": aug}}
               for i,aug in enumerate(["std", "reverse", "hflip", "reverse_hflip"])]
    stats = adapter.compute_statistics(results)
    assert stats["robust_accuracy"] == {"1": 1, "2": 1, "3": 1, "4": 0}
    assert adapter.compute_statistics(results[:3])["robust_accuracy"] is None


def test_vlm4d_duplicate_correct_option_is_preserved():
    adapter = DynamicDataset("VLM4D", "unused")
    samples = adapter.prepare_dataset([{"id": "validation_633", "question": "Moving how?",
        "choices": {"A": "not sure", "B": "not moving", "C": "away", "D": "away"}, "answer": "away"}])
    assert samples[0]["answer"] == ["C", "D"]
    assert adapter.evaluate_results(samples, ["D"])[0]["is_correct"]


def test_dynamic_strata_do_not_leak_to_public_input():
    adapter = DynamicDataset("VLM4D", "unused", split="synthetic_mc")
    sample = adapter.prepare_dataset([{"id": "validation_1", "question": "Does it move?",
                                     "choices": {"A": "yes", "B": "no"}, "answer": "no"}])[0]
    assert sample["metadata"]["scoring_stratum"] == "synthetic_false_positive"
    example = {"sample": sample, "sample_id": "vlm4d:0", "source_id": "validation_1", "aliases": [],
               "question": sample["question"], "task": "multiple-choice"}
    assert "false_positive" not in str(media_manifest(example))
    assert "false_positive" not in sample["question"]


def test_dsi_category_names_from_official_schema():
    adapter = DynamicDataset("DSI-Bench", "unused", split="std")
    sample = adapter.prepare_dataset([{"question": "Motion?", "options": "A: up; B: down", "GT": "A",
                                     "cate": "1", "relative_path": "CameraBench/clip.mp4"}])[0]
    assert sample["metadata"]["question_type"] == "object_motion_moving_camera"
    assert sample["video"] == "CameraBench/clip.mp4"


def test_manifest_does_not_export_labels(tmp_path):
    args = settings(tmp_path)
    session = BenchmarkSession(get_spec("VLM4D"), data=args.data)
    record = media_manifest(next(session.batches())[0])
    assert "answer" not in record and "ground_truth" not in record
    assert record["sample_id"] == "vlm4d:0"


def test_media_exports_lazy_frames_without_inventing_time(tmp_path):
    from core.media import export_media
    from datasets import Dataset
    stream = io.BytesIO(); Image.new("RGB", (3, 2)).save(stream, format="PNG")
    dataset = Dataset.from_dict({"images": [[{"bytes": stream.getvalue(), "path": None}]]})
    value = {"__lazy_hf_images__": True, "dataset": dataset, "column": "images", "row_index": 0}
    exported = export_media(value, tmp_path)
    assert Image.open(exported[0]["path"]).size == (3, 2)
    assert "timestamp" not in exported[0]
    with pytest.raises(FileExistsError):
        export_media(value, tmp_path)


def test_replay_missing_samples_penalized_and_resume_verified(tmp_path):
    predictions = tmp_path / "predictions.json"
    write_json(predictions, {"q1": "A"})
    args = settings(tmp_path, predictions=str(predictions))
    result = run(args)
    assert result["primary_metric"]["score_100"] == 50
    assert result["status_counts"] == {"ok": 1, "missing_prediction": 1}
    args.resume = True
    assert run(args)["reused_predictions"] == 1
    args.temperature = .1
    with pytest.raises(ValueError, match="configuration mismatch"):
        run(args)


def test_truncated_completion_not_scored_as_correct(tmp_path):
    predictions = tmp_path / "predictions.json"
    write_json(predictions, [{"id": "q1", "raw_output": "A", "finish_reason": "length"}, {"id": "q2", "raw_output": "B"}])
    result = run(settings(tmp_path, predictions=str(predictions)))
    assert result["status_counts"]["invalid_completion"] == 1
    assert result["primary_metric"]["score_100"] == 50


def test_legacy_vsi_replay_uses_qa_index_and_checks_identity(tmp_path):
    from core.inference import ReplayEngine
    path = tmp_path / "predictions.json"
    write_json(path, [{"id": "vsibench_2929", "qa_index": 2929, "question": "Which?", "scene_name": "room",
                       "options": ["A. chair"], "pred_answer": "A"}])
    engine = ReplayEngine(path)
    meta = {"scene_name": "room", "options": ["A. chair"]}
    assert engine.infer("vsi_bench:2929", 2947, question="Which?\nA. chair", metadata=meta)["raw_output"] == "A"
    with pytest.raises(ValueError, match="no longer matches"):
        engine.infer("vsi_bench:2929", 2947, question="Changed?", metadata=meta)


def test_http_text_prompt_and_resume(tmp_path):
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            assert self.path == "/v1/chat/completions"
            output = "B" if "stayed still" in request["messages"][-1]["content"] else "A"
            body = json.dumps({"choices": [{"message": {"content": output}, "finish_reason": "stop"}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        args = settings(tmp_path, base_url=f"http://127.0.0.1:{server.server_port}/v1")
        assert run(args)["primary_metric"]["score_100"] == 100
        args.resume = True
        assert run(args)["reused_predictions"] == 2
        assert len(requests) == 2
        for req in requests:
            assert req["model"] == "test-llm" and req["temperature"] == 0
            assert all(isinstance(m["content"], str) for m in req["messages"])
            assert "ground_truth" not in json.dumps(req)
    finally:
        server.shutdown(); server.server_close(); worker.join()


def test_journal_recovers_only_partial_final_line(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text('{"sample_id":"1"}\n{"sample_id":')
    assert list(read_journal(path)) == ["1"]
    assert path.read_text() == '{"sample_id":"1"}\n'
    path.write_text('{"sample_id":"2"}')
    assert list(read_journal(path)) == ["2"] and path.read_text().endswith("\n")
    path.write_text('{"sample_id":\n{"sample_id":"2"}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_journal(path)


def test_3dsr_limit_keeps_circular_group(tmp_path):
    stream = io.BytesIO(); Image.new("RGB", (2, 2)).save(stream, format="PNG")
    encoded = base64.b64encode(stream.getvalue()).decode()
    path = tmp_path / "data.tsv"
    rows = [{"index": str(i), "qid": f"q1-{i}", "category": "height_higher", "question": "Which is higher?",
             "answer": "A", "A": "chair", "B": "table", "image": encoded} for i in range(4)]
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); writer.writeheader(); writer.writerows(rows)
    session = BenchmarkSession(get_spec("3DSRBench"), data=str(path), limit=1, batch_size=2)
    assert session.selected_count == 4
    scored = []
    for batch in session.batches():
        scored += session.adapter.evaluate_results([e["sample"] for e in batch], ["A", "B"])
    stats = session.adapter.compute_statistics(scored)
    assert stats["overall_accuracy"] == 0 and stats["row_accuracy"] == .5


def test_trajectory_score_direction_and_coverage():
    metric = primary_metric(get_spec("ShareRobot-Traj"), {"normalized_rmse_score": 15, "valid_samples": 1, "total_samples": 2})
    assert metric["score_100"] == 85 and metric["valid_samples"] == 1


def test_project_vsi_micro_vs_physbrain_macro():
    from benchmark.loader import LegacyVSI
    from benchmark.project_vsi import ProjectVSI
    native = make_adapter(get_spec("VSI-Bench"), "unused", "train", "test")
    rows = [{"id": str(i), "question": "Test", "ground_truth": "A", "question_type": task} for i,task in
            enumerate(["route_planning", "route_planning", "object_rel_distance"])]
    adapter = LegacyVSI(native)
    samples = adapter.prepare_dataset(rows)
    outputs = ["A", "A", "B"]
    assert adapter.compute_statistics(adapter.evaluate_results(samples, outputs))["overall_score"] == .5
    project = ProjectVSI(adapter)
    assert project.compute_statistics(project.evaluate_results(samples, outputs))["overall_score"] == pytest.approx(2/3)


def test_qspatial_requires_units_and_uses_strict_factor_two():
    adapter = make_adapter(get_spec("Q-Spatial-Bench"), "unused", "QSpatial_plus", "test")
    sample = {"question": "Distance?", "metadata": {"idx": 0, "question_id": "q", "question_type": "distance",
               "answer_value": 100, "answer_unit": "cm"}}
    scored = adapter.evaluate_results([sample]*3, ["1 m", "2 m", "100"])
    assert [row["success"] for row in scored] == [True, False, False]


def test_point_metric_penalizes_missing_output():
    adapter = make_adapter(get_spec("PointBench"), "unused", "train", "test")
    sample = {"question": "Point to the object.", "metadata": {"idx": 0, "question_id": "q", "category": "pointing",
               "width": 100, "height": 100, "mask": Image.new("L", (100,100), 255), "expected_count": 1}}
    results = adapter.evaluate_results([sample, sample], ["[[500, 500]]", ""])
    stats = adapter.compute_statistics(results)
    assert stats["non_strict_micro_f1"] == pytest.approx(2/3)
    assert primary_metric(get_spec("PointBench"), stats)["score_100"] == pytest.approx(200/3)


def test_mask_change_changes_resume_fingerprint():
    from benchmark.loader import scoring_fingerprint
    a = {"answer": None, "metadata": {"mask": Image.new("L", (2, 2), 0)}}
    b = {"answer": None, "metadata": {"mask": Image.new("L", (2, 2), 255)}}
    assert scoring_fingerprint(a) != scoring_fingerprint(b)


def test_nonrigid_camera_rejected():
    with pytest.raises(ValueError, match="rigid"):
        normalize_scene(scene(cameras=[{"camera_to_world": [[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]))


def test_vendored_protocol_files_only_have_declared_import_rewrites():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root / "docs/upstream_manifest.json")
    for entry in manifest["files"]:
        data = (root / entry["local_path"]).read_bytes()
        if entry["local_path"].endswith(".py"):
            data = data.replace(b"from core.physbrain.", b"from core.")
        assert hashlib.sha256(data).hexdigest() == entry["upstream_sha256"], entry["local_path"]
