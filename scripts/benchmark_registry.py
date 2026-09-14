"""Benchmark data/scoring registry. Imports no GPU or dataset packages."""
from __future__ import annotations

from dataclasses import dataclass

UPSTREAM_REVISION = "4b37ca2f184bd76b98827481a6739f89ea06d730"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    module: str
    class_name: str
    dataset: str
    split: str
    group: str
    score_key: str
    score_scale: float = 100.0
    priority: str = "extension"

    @property
    def slug(self):
        return self.name.lower().replace("-", "_")


_ROWS = [
    ("BLINK", "blink", "BLINKDataset", "BLINK-Benchmark/BLINK", "val", "static", "overall_accuracy", 100, "core"),
    ("CV-Bench", "cvbench", "CVBenchDataset", "nyu-visionx/CV-Bench", "test", "static", "overall_accuracy", 100, "core"),
    ("3DSRBench", "threedsrbench", "ThreeDSRBenchDataset", "VLyb/3DSRBench", "test", "static", "overall_accuracy", 100, "core"),
    ("EmbSpatial-Bench", "embspatial", "EmbSpatialDataset", "FlagEval/EmbSpatial-Bench", "test", "static", "overall_accuracy", 100, "core"),
    ("MindCube", "mindcube", "MindCubeDataset", "VLyb/MindCube-TinyBench", "tinybench", "multiview", "overall_accuracy", 100, "core"),
    ("MMSI-Bench", "mmsi_bench", "MMSIBenchDataset", "RunsenXu/MMSI-Bench", "test", "multiview", "overall_accuracy", 100, "core"),
    ("Q-Spatial-Bench", "qspatial", "QSpatialBenchDataset", "andrewliao11/Q-Spatial-Bench", "QSpatial_plus", "static", "success_rate", 100, "core"),
    ("RoboSpatial-Home", "robospatial", "RoboSpatialDataset", "chanhee-luke/RoboSpatial-Home", "context+compatibility+configuration", "grounding", "non_strict_overall_score", 100, "extension"),
    ("SAT", "sat", "SATDataset", "FlagEval/SAT", "test", "action_conditioned", "overall_accuracy", 100, "core"),
    ("VSI-Bench", "vsibench", "VSIBenchDataset", "IffYuan/vsi-bench", "train", "static_video", "overall_score", 100, "core"),
    ("ViewSpatial-Bench", "viewspatial", "ViewSpatialDataset", "lidingm/ViewSpatial-Bench", "test", "multiview", "overall_accuracy", 100, "core"),
    ("COSMOS", "cosmos", "COSMOSDataset", "IffYuan/COSMOS", "train", "video_reasoning", "overall_score", 100, "extension"),
    ("EgoPlan-Bench2", "egoplan2", "EgoPlan2Dataset", "IffYuan/ego-plan", "train", "video_reasoning", "overall_score", 100, "extension"),
    ("ERQA", "erqa", "ERQADataset", "FlagEval/ERQA", "test", "embodied", "overall_accuracy", 100, "extension"),
    ("ERQA-PLUS", "erqa_plus", "ERQAPlusDataset", "huggingdas/erqa-plus", "train", "embodied", "overall_accuracy", 100, "extension"),
    ("RoboVQA", "robovqa", "RoboVQADataset", "VLyb/RoboVQA-16frames", "train", "video_reasoning", "overall_bleu", 1, "extension"),
    ("VLABench", "vlabench", "VLABenchDataset", "VLyb/VLABench", "local", "planning", "overall_score", 1, "extension"),
    ("Part-Affordance", "partafford", "PartAffordDataset", "IffYuan/Part-Affordance-2K", "train", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("PIOBench", "pio", "PIOBenchDataset", "IffYuan/PIO-Bench", "train", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("PixMo-Points", "pixmo_points", "PixmoPointsDataset", "IffYuan/pixmo-points-eval", "train", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("PointBench", "pointbench", "PointBenchDataset", "IffYuan/PointBench", "train", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("RefSpatial-Bench", "refspatial", "RefSpatialBenchDataset", "BAAI/RefSpatial-Bench", "location+placement+unseen", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("RoboAfford", "roboafford", "RoboAffordDataset", "Zray26/roboafford-eval", "test", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("RoboRefit", "roborefit", "RoboRefitDataset", "VLyb/RoboRefit-corrected", "test", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("VABench-Point", "vabench_point", "VABenchPointDataset", "IffYuan/VABench-P", "test", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("Where2Place", "where2place", "Where2PlaceDataset", "FlagEval/Where2Place", "test", "grounding", "non_strict_micro_f1", 100, "extension"),
    ("ShareRobot-Traj", "sharerobot_bench_trajectory", "SharerobotTraceDataset", "IffYuan/sharerobot_trajectory", "train", "trajectory_generation", "normalized_rmse_score", 1, "extension"),
    ("VABench-V-Trace", "vabench_visual_trace", "VABenchVisualTraceDataset", "IffYuan/vabench-v", "train", "trajectory_generation", "normalized_rmse_score", 1, "extension"),
    ("STI-Bench", "dynamic", "DynamicDataset", "MINT-SJTU/STI-Bench", "train", "dynamic", "overall_accuracy", 100, "core"),
    ("VLM4D", "dynamic", "DynamicDataset", "shijiezhou/VLM4D", "real_mc", "dynamic", "overall_accuracy", 100, "core"),
    ("DSI-Bench", "dynamic", "DynamicDataset", "Viglong/DSI-Bench", "std", "dynamic", "overall_accuracy", 100, "extension"),
]
BENCHMARKS = {row[0]: BenchmarkSpec(*row) for row in _ROWS}
ALIASES = {"vsibench": "VSI-Bench", "qspatial": "Q-Spatial-Bench",
           "embspatial": "EmbSpatial-Bench", "robospatial": "RoboSpatial-Home",
           "egoplan2": "EgoPlan-Bench2", "viewspatial": "ViewSpatial-Bench",
           "part-affordance-2k": "Part-Affordance", "sharerobot-trajectory": "ShareRobot-Traj",
           "vabench-visual-trace": "VABench-V-Trace"}


def get_spec(name):
    name = ALIASES.get(name.lower(), name)
    for spec in BENCHMARKS.values():
        if name.lower().replace("_", "-") == spec.name.lower().replace("_", "-"):
            return spec
    raise ValueError(f"Unknown benchmark {name!r}. Run python eval.py --list-benchmarks")
