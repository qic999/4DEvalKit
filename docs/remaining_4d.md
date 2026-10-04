# Remaining 4D benchmarks

Checked on 2026-10-03. **Not ready** still means a complete model evaluation has
not been validated. Existing scorer wrappers alone do not change that status.

| Benchmark | What can be resolved | Remaining input or limitation |
|---|---|---|
| [V-STaR](vstar.md) | Added data preparation, all five conditioned subtasks, three observation modes, native 72B judging and joint scoring. Public videos and judge are downloading; a real two-question pilot is queued. | Validate inference and judging, then run the full split. |
| [MVVBench](../scripts/prepare_mvvbench.py) | Added an exact required-view inventory, source linking and strict preparation. | 1,940 of 1,943 required view files are absent locally. The official mapping also lacks `upenn_0714_Cooking_6_3_view5.mp4`, referenced by one question. Obtain source videos and resolve that upstream mapping gap. |
| [ADT](additional_4d_evaluation.md#perception-and-grounding-scorers) | Official numerical scoring is integrated; a native-pose exporter can be added once object-frame conventions are specified. | Held-out GT and object prototypes, plus semantic object-frame alignment. Current local ADT data are training sequences and cannot stand in for the held-out evaluation. |
| [HOI4D Object Tracking](https://github.com/hoi4d/ObjTracking) | Add the published submission-format exporter once the tracking test package is available. | Test data and pose-format integration. The inspected official repository supplies preprocessing/submission instructions, not a local tracking scorer. |
| [HOT3D](https://github.com/facebookresearch/hot3d/blob/main/hot3d/clips/README.md) | Native BOP object-pose export and environment setup are engineering work. | CAD/object-frame alignment; test object GT is not public. A train-split diagnostic is not the official held-out test result. |
| [nuScenes Tracking](https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk/nuscenes/eval/tracking) | Add a calibrated, causal tracker/exporter and use the official validation scorer. | Full tracking data and metadata, global coordinates, semantic classes, persistent IDs, velocity and detection of later entrants. First-frame object slots are insufficient. |
| [TAPVid-3D](https://tapvid3d.github.io/) | The scorer is integrated; a queried-point predictor can be connected. | The current encoder predicts boxes. Arbitrary surface-point trajectories and visibility require an additional model capability, not a box-format conversion. |
| [DA4D / DetAny4D](https://jarvishou829.github.io/DA4D/) | Integrate when its evaluation package is released. | Official project page still advertises `Code(Soon)`; a runnable public evaluation package was not found. |
| [MLLM4D-Bench](https://huggingface.co/datasets/flow666/MLLM-4D-Datasets/tree/main) | Integrate when independent QA labels and split are available. | Inspected test archive contains videos, without independent QA annotations. Reusing VLM4D questions would not reproduce this benchmark. |

ADT and HOI4D retain the owner-confirmed
[training/test assumptions and exclusions](adt_hoi4d_split_protocol.md).

## MVVBench data inventory

The full mapping lists 3,082 videos, but the questions reference only 1,943
distinct views. Preparation requires exactly those views and retains every
question; it neither downloads unused cameras nor silently drops missing ones.

| Source | Required missing views |
|---|---:|
| EgoExo4D | 1,595 |
| MMPTrack | 212 |
| Panoptic | 132 |
| Missing from the upstream mapping | 1 |

Three Panoptic view files are present, enough for the previously reported two
questions. [EgoExo4D access](https://docs.ego-exo4d-data.org/getting-started/)
requires its data license and authorized download credentials. Use existing
authorized local datasets with the source-directory layout in the
[official preparation script](https://huggingface.co/datasets/everex/MVVBench/blob/main/prepare_videos.py):

```bash
python -m scripts.prepare_mvvbench \
  --annotations data/MVVBench/all_questions.json \
  --mapping data/MVVBench/video_mapping.csv \
  --videos data/MVVBench/videos --output results/mvv_inventory \
  --source-root /path/to/source_datasets --link --prepare
```

Without `--source-root --link --prepare`, this only writes an inventory and
`missing_videos.csv`. `--source panoptic` selects **all** 95 Panoptic questions
explicitly; it still fails preparation if any required view is missing. A
source-subset result must be labeled as such and is not the full MVVBench score.
