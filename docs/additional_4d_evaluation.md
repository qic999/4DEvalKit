# Additional 4D evaluation workflows

Run commands from the repository root. Install `requirements-4d.txt` for the
numerical scorers and Physion readout. Encoder and vLLM environments remain
separate. See the [runtime report](../reports/additional_4d_20261002/comparison.md)
for actual executed subsets and full-run status.

## Temporal QA

| `--benchmark` | Annotation input | Video root / protocol |
|---|---|---|
| MotionBench | `MotionBench/video_info.meta.jsonl` | Both `MotionBench/public-dataset` and `MotionBench/self-collected`; only the 4,018 labeled DEV QA records. `NA` answers belong to hidden TEST. |
| TempCompass | `multi-choice/test-00000-of-00001.parquet` | Extract `tempcompass_videos.zip`; 1,580 MC questions and 410 videos. Upstream `disable_llm` answer matching, overall and coarse-aspect accuracy. |
| CLEVRER | Official `questions/validation.json` | All five extracted validation video directories; 5,000 videos. Every MC candidate is a correct/wrong decision; all candidates must be right for per-question accuracy. Descriptive answers use the public word/count vocabulary. |
| 4D-Bench | `4D_Object_Question_Answering/data/4d_qa.json` | `4d_object_multi_view_videos`; 751 QA records. Views 1, 8, 16 with six frames per view match the inspected official example. |
| MVVBench | `all_questions.json` | Videos mapped by the release's `video_mapping.csv` / `prepare_videos.py`. Preserve ordered camera IDs. GT relevant-interval annotations are not used to crop inputs. Local exact-choice scoring includes category and static/dynamic breakdowns. |

Download MotionBench DEV with its pinned release:

```bash
python -m scripts.download_motionbench --output data/MotionBench
python -m scripts.prepare_additional_4d --benchmark MotionBench \
  --annotations data/MotionBench/MotionBench/video_info.meta.jsonl \
  --video-root data/MotionBench/MotionBench/public-dataset \
               data/MotionBench/MotionBench/self-collected \
  --output data/prepared/MotionBench
```

For other benchmarks, use their public download instructions in the
[official-code audit](official_4d_evaluators.md), then run the same preparation
command with the annotation file and video roots in the table. Missing videos
fail preparation; they are never silently excluded. `--limit N` creates an
explicit source-order smoke subset. CLEVRER expands the selection to whole
question groups. Prepared manifests contain public observations and questions;
scoring labels stay in the separate annotation file.

Copy [the example config](../configs/additional_4d.example.json), set local
checkpoint/environment paths and prepared datasets, and run:

```bash
mkdir -p logs
nohup python -u -m scripts.run_additional_4d \
  --config configs/my_additional_4d.json --output results/additional_4d \
  > logs/additional_4d.log 2>&1 < /dev/null &
```

The runner generates GroundingDINO proposals, runs the checkpoint, then evaluates
`boxes`, `rgb`, and `rgb_boxes` with the same reasoner and public observations.
`encoding_shards` partitions complete video groups across GPU workers. Identical
observations can reuse predictions across questions. GPU leases avoid stopping
other toolkit tasks. Each run writes `status.json`, logs, prediction journals,
geometry and final scores; rerun with unchanged configuration to resume.

For multiview videos, `video_frames` means frames **per camera**. Choose a total
within the server's 32-image limit (for example, four frames for six cameras).
Geometry is inferred independently for each camera video. IDs and timestamps
are preserved; no shared world alignment or cross-camera object identity is
invented. `--require-tracks` applies to single-camera trajectory bundles; the
multicamera format uses timestamped objects in `views`.

The geometry initialization uses predicted boxes in each camera's first frame.
Later entrants may be missed. These QA experiments do not establish open-world
multi-object tracking performance.

CLEVRER outputs descriptive and per-option/per-question category scores.
`overall_question_accuracy` is a convenience aggregate across question types;
compare the category metrics, not this aggregate, with published CLEVRER tables.

## Physion OCP box-feature readout

This workflow evaluates frozen **predicted box features** with the official
logistic readout, separately from LLM reasoning. Its OCP observations follow the
official `frame_gap=150` schedule: frames 0/15/30/45, or 0/0/0/15 for collisions.
Repeated initial observations are reproduced in the exported feature tensor.
No future contact time or test label selects encoder frames. OCD is not included.

1. Download the official [MP4 archive](https://storage.googleapis.com/physion-dataset/physion_dataset.zip)
   and the pinned [loader source](https://github.com/neuroailab/physion_evaluator/blob/03924d4a4c518044a279c10f286335bb11a1a709/physion_evaluator/dataloader_mp4s.py).
2. Prepare observations and separate scoring metadata:

   ```bash
   python -m scripts.prepare_physion --data-root data/Physion \
     --archive data/Physion/physion_dataset.zip --output data/prepared/Physion \
     --upstream-loader external/physion_evaluator/physion_evaluator/dataloader_mp4s.py
   ```

   For a smoke run, add `--train-per-scenario 10 --test-per-scenario 3`.
3. Set `modes: []` in the suite configuration and use two jobs named
   `Physion-train` and `Physion-test`, pointing to prepared `train` / `test`
   directories. Then run:

   ```bash
   python -m scripts.run_physion_box_readout --config configs/my_physion.json \
     --output results/physion
   ```

Each frame pools center, dimensions, quaternion and confidence using mean, std,
min and max, plus object count (45 values). The official training script fits
normalization and selects readout regularization on training data only, then
reports per-scenario test accuracy. Smoke subset results are not full Physion
scores or evaluations of an unspecified hidden-layer representation.

## Perception and grounding scorers

`scripts.score_official_4d` downloads hash-verified sources from the
[pinned audit manifest](../reports/upstream_eval_audit_20261002/manifest.json).
Numerical function bodies are used unchanged. ADT visualization/C++ imports and
V-STaR's eager 72B judge initialization are excluded from numerical-only calls.

```bash
python -m scripts.verify_official_4d --output results/scorer_fixtures
python -m scripts.score_official_4d --benchmark ADT \
  --config configs/my_adt.json --output results/adt.json
```

Configuration fields:

| Scorer | Required config fields |
|---|---|
| ADT | `annotations` ZIP, `predictions` ZIP, `sequences` list, `prototypes` list. Archives follow the official timed object-model-pose format. Box rotations need declared object-frame alignment before export. The pinned multi-candidate association behavior is preserved. |
| TAPVid-3D | `annotations` NPZ (`tracks`, `occluded`, `intrinsics_params`, optional `query_points`), `predictions` NPZ (`tracks`, `occluded`), explicit `scaling`, optional `order`. Requires actual queried-point predictions, not box centers. |
| V-STaR-grounding | `annotations` native QA JSON, `predictions` mapping every annotation index to `temporal` interval and `spatial` timestamp-to-box mapping. Reports numerical grounding only; the official joint score additionally needs semantic answers and its Qwen2.5-72B judge. |
| Physion | Native official readout arguments; generated automatically by the Physion workflow above. |

For HOT3D / nuScenes, `scripts.run_external_4d` invokes an external official
checkout without changing its evaluator:

```bash
python -m scripts.run_external_4d --benchmark nuScenes-Tracking \
  --config configs/my_nuscenes_tracking.json --output results/nuscenes_tracking
```

Config requires `upstream_repo`, `python`, `data_root`, and native `predictions`.
nuScenes also accepts `version` and `split`; HOT3D-BOP requires
`targets_filename`. Checkouts must match the commits in
`scripts/run_external_4d.py`. Install each upstream project's own dependencies.
The launcher alone does not generate semantic object-model poses or globally
calibrated online tracks from a geometry bundle.

The local ADT cache covers training sequences; held-out challenge GT is still
needed. The inspected HOT3D test clip has camera parameters and images but no
object GT. HOI4D's official task repository exposes a submission interface, not
a local object-tracking scorer. MLLM-4D's inspected test ZIP contains videos but
no independent QA labels. DA4D still needs a usable evaluation release. These
remaining requirements are recorded individually in the runtime report.

Video decoding uses an explicit PyAV fallback when OpenCV random seeking fails. It samples actual decoded frames and PTS rather than phantom container frame indices. Install the root requirements in all three environments (toolkit, detector and encoder) so this fallback is available consistently. Decoder errors still fail the sample.
