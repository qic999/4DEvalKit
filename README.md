# 4DEvalKit

Evaluate spatial perception and reasoning on images, multiple views, and video.
Choose a workflow below to score 3D boxes directly or answer benchmark questions
from RGB, geometry, captions, or combinations of these inputs.

Use Python 3.10 or newer for the toolkit and run commands from the repository root:

```bash
git clone https://github.com/qic999/4DEvalKit.git
cd 4DEvalKit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model inference and official perception scoring use separate environments.
The dataset download helpers require Python 3.12 or newer.
See the [usage guide](docs/usage.md) for input preparation, background suites,
resume, offline scoring, and tests.

## 3D Perception

Predict metric, oriented 3D boxes from an RGB image and ground-truth 2D box
prompts, then score them with the pinned WildDet3D evaluator. This workflow
does not use a language model. GT category IDs are attached for scoring;
category names, GT depth, camera intrinsics, camera poses, and GT 3D boxes are
not supplied to the encoder.

| Dataset | Split | Main metric |
|---|---|---|
| ScanNet | 3D-MOOD validation | Canonical ODS |
| Argoverse 2 | 3D-MOOD validation | Canonical ODS |
| Omni3D | KITTI, nuScenes, SUNRGBD, Hypersim, ARKitScenes, Objectron test splits | Oriented 3D IoU AP |
| WildDet3D-Bench | InTheWild_v3_val | Center-distance AP |

These are box-conditioned 3D regression evaluations. Outdoor driving images
also contain moving objects; single-frame scores measure their geometry, not
their motion over time.

Follow the [perception setup guide](docs/perception_evaluation.md) to obtain the
annotations and original RGB files and install the official scorer. Set the
Python executables for your encoder and scorer environments:

```bash
export ENCODER_PYTHON=/path/to/encoder/env/bin/python
export SCORER_PYTHON=/path/to/wilddet3d/env/bin/python

git clone https://github.com/allenai/WildDet3D.git external/WildDet3D
git -C external/WildDet3D checkout 1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a

$SCORER_PYTHON -m scripts.prepare_wilddet3d_perception \
  --upstream external/WildDet3D \
  --annotation data/scannet/annotations/ScanNet_val.json \
  --data-root data/scannet --benchmark scannet \
  --output results/perception/scannet_manifest.json

CUDA_VISIBLE_DEVICES=0 $ENCODER_PYTHON -m scripts.encode_wilddet3d_perception \
  --manifest results/perception/scannet_manifest.json \
  --model-repo /path/to/spatial_encoder_v2_small \
  --checkpoint /path/to/checkpoint.pt --profile lite \
  --resolution 518 --spatial-resolution 518 --model-image-size 518 \
  --output results/perception/scannet/small_e100

$SCORER_PYTHON -m scripts.score_wilddet3d_perception \
  --upstream external/WildDet3D \
  --manifest results/perception/scannet_manifest.json \
  --predictions results/perception/scannet/small_e100 \
  --output results/perception/scannet/small_e100/metrics.json \
  --metrics dist bbox
```

Match input sizes to the checkpoint: WDS518 e100 uses `518 / 518 / 518`;
historical Full e64 and Small e34 use `1024 / 504 / 1008`. Full uses
`--profile full`; both Small models use `--profile lite`.

For a complete background evaluation, edit the
[perception suite config](configs/perception_evaluation.example.json), prepare
its manifests, and run:

```bash
mkdir -p logs results/perception
cp configs/perception_comparison.example.json results/perception/comparison_config.json
nohup python -u -m scripts.run_wilddet3d_perception \
  --config configs/my_perception.json > logs/perception.log 2>&1 < /dev/null &
python -m scripts.summarize_wilddet3d_perception --run-root results/perception
```

The [completed Full / Old Small / WDS518 e100 comparison](reports/wilddet3d_perception_20261002/comparison.md)
includes individual datasets and the official joint Omni3D score, with
[CSV](reports/wilddet3d_perception_20261002/comparison.csv) and
[protocol metadata](reports/wilddet3d_perception_20261002/protocol.json).
WildDet3D-Bench results use an explicitly recorded **2460/2470-image subset**.
Paper values remain references where prompt modes, training data, or subsets differ.

## 3D Reasoning

Evaluate spatial questions about objects, distances, directions, layout, and
relationships across views. Supported benchmarks include BLINK, CV-Bench,
3DSRBench, EmbSpatial-Bench, Q-Spatial-Bench, MindCube, MMSI-Bench,
ViewSpatial-Bench, VSI-Bench, and SAT. Their question-answering scores are
separate from the direct box metrics above.

Export a benchmark's public media manifest, then generate matching geometry
with your encoder or convert existing boxes using the
[geometry schema](docs/geometry_schema.md). A static scene observed in a video
can still be a 3D spatial reasoning task.

```bash
python eval_vsi_bench.py \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --geometry data/geometry/vsi_bench.json \
  --dry-run --output results/vsi_check.json

python eval_vsi_bench.py \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --geometry data/geometry/vsi_bench.json \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 0 --max-tokens 4096 \
  --output results/vsi_scannet.json --resume
```

The endpoint must support `/v1/chat/completions`. Geometry-only and caption-only
inputs can use a text model; RGB inputs require a vision-capable model. Offline
scoring uses `--predictions saved_responses.json` instead of a live endpoint.
Keep the VSI scoring protocol fixed across runs; the default is
`--vsi-metric-protocol physbrain`.

For controlled input comparisons, use `--observation-mode boxes`, `rgb`,
`rgb_boxes`, `caption`, or `caption_boxes` with `--media-manifest`. Caption modes
also need `--captions`. The guides cover
[RGB and geometry comparisons](docs/rgb_evaluation.md#native-answer-formats-and-matched-observation-ablations),
[caption comparisons](docs/caption_evaluation.md), and
[evaluating a new checkpoint](docs/checkpoint_evaluation.md).
On a compatible vLLM server, `--answer-format native` constrains supported
tasks to the expected answer format; inspect completion status before comparing scores.

Object boxes provide geometric information. Appearance, fine-grained parts,
contact, and robot state may require additional observations. See the
[benchmark matrix](docs/benchmark_matrix.md) for scope and extensions.

## 4D Perception

The current dynamic-scene perception evaluation uses **Stereo4D: 383 independent
frames**, scored with center-distance AP and rare/common/frequent category AP.
It measures per-frame 3D geometry in dynamic scenes. Cross-frame identity,
trajectory accuracy, and temporal consistency metrics are **not implemented**.

Download the original released images and prepare the manifest:

```bash
python -m scripts.prepare_wilddet3d_stereo --output data/stereo4d

$SCORER_PYTHON -m scripts.prepare_wilddet3d_perception \
  --upstream external/WildDet3D \
  --annotation data/stereo4d/annotations/Stereo4D_val.json \
  --data-root data/stereo4d --benchmark stereo4d \
  --output results/perception/stereo4d_manifest.json
```

Run `scripts.encode_wilddet3d_perception` and
`scripts.score_wilddet3d_perception` as above, using this manifest and separate
`results/perception/stereo4d/<model>/` output directories. Select `--metrics dist`
for the main score. The [perception comparison](reports/wilddet3d_perception_20261002/comparison.md)
reports this frame-level result explicitly.

The [RGB video pipeline](docs/rgb_evaluation.md#inputs-and-geometry) also exports
timestamped boxes, stable object slots, and predicted camera poses for downstream
reasoning. Exporting tracks is distinct from evaluating them against temporal
ground truth. Its first-frame proposal strategy can miss objects entering later.

## 4D Reasoning

Evaluate object motion, temporal order, changing spatial relationships, and
camera motion with STI-Bench, VLM4D, and DSI-Bench. Geometry inputs should contain
`tracks[].observations[]` with stable IDs, timestamps in seconds, and per-frame
boxes in a consistent coordinate frame.

```bash
python eval_sti_bench.py --data /path/to/STI-Bench/qa.parquet \
  --geometry data/geometry/sti_bench.json --require-tracks \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --max-tokens 4096 --output results/sti_bench.json --resume

python eval_vlm4d.py --split real_mc \
  --geometry data/geometry/vlm4d_real.json --require-tracks \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --max-tokens 4096 --output results/vlm4d_real.json --resume
```

Run VLM4D's `real_mc` and `synthetic_mc` splits separately. DSI-Bench supports
`--split std` or `--split all`; each video augmentation needs matching geometry.
`--require-tracks` checks track presence; it does not calculate tracking accuracy.
Use the same sampled frames, timestamps, input mode, and reasoning settings
when comparing encoders. See [scoring protocols](docs/metric_protocols.md) for
the direct-choice and augmentation aggregation definitions.

Try the included synthetic fixtures without model weights or a server:

```bash
python eval_vlm4d.py --data examples/dynamic_qa.json \
  --geometry examples/dynamic_geometry.json --require-tracks \
  --dry-run --output results/example_check.json

python eval_vlm4d.py --data examples/dynamic_qa.json \
  --predictions examples/dynamic_predictions.json \
  --model synthetic_fixture --output results/example_replay.json
```

For background execution across benchmarks and models, use
[the suite config](configs/core_suite.example.json) and the
[suite instructions](docs/usage.md#run-multiple-benchmarks-and-models).
Each QA evaluation saves scores, an incremental response journal, and input
fingerprints. Resume with unchanged inputs using `--resume`.

[Geometry format](docs/geometry_schema.md) ·
[Scoring protocols](docs/metric_protocols.md) ·
[Third-party notices](THIRD_PARTY_NOTICES.md)
