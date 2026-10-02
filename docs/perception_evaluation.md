# Evaluate 3D box perception

This workflow scores SpatialEncoder's oriented 3D boxes directly, without a
language model. It follows WildDet3D's RGB + ground-truth 2D box prompt setting.
Ground-truth category IDs identify prompted objects for scoring; category names,
depth, camera intrinsics, poses, and 3D annotations are not encoder inputs.

Completed scores for Full e64, Old Small e34, and Small WDS518 e100 are in the
[published comparison](../reports/wilddet3d_perception_20261002/comparison.md).
The toolkit uses Python 3.10+; the optional dataset download helpers use Python
3.12+. Run GPU inference and official scoring in their respective environments.

## Datasets and metrics

| Dataset | Evaluation split | Primary metric |
|---|---|---|
| WildDet3D-Bench | InTheWild_v3_val | Center-distance AP; rare/common/frequent AP |
| Stereo4D | Stereo4D_val, 383 images | Center-distance AP; rare/common/frequent AP |
| ScanNet | 3D-MOOD ScanNet_val, 6,240 images | ODS, AP, normalized ATE/ASE/AOE |
| Argoverse 2 | 3D-MOOD Argoverse_val, 4,806 images | ODS, AP, normalized ATE/ASE/AOE |
| Omni3D | KITTI, nuScenes, SUNRGBD, Hypersim, ARKitScenes, Objectron **test** splits | Oriented 3D IoU AP, thresholds 0.05–0.50 |

Stereo4D's released perception split contains independent images from videos;
this evaluation does not measure tracking. Box prompts isolate 3D regression
and do not evaluate the discovery of objects or categories. Scores from the
older ScanNet val88 merged-scene AABB protocol are a separate measurement.

## Set up the official scorer

Keep the encoder's existing environment and a separate environment satisfying
[WildDet3D's requirements](https://github.com/allenai/WildDet3D#installation),
including `vis4d==1.0.0` and `vis4d_cuda_ops`. The scoring adapter loads only the
official dataset/metric modules, so its SAM3 and depth models are unnecessary.

```bash
git clone https://github.com/allenai/WildDet3D.git external/WildDet3D
git -C external/WildDet3D checkout 1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a
```

The commit is checked by the adapter. Annotation filters, ignore regions,
category mappings, matching thresholds, frequency groups, and error aggregation
come from the official implementation. `eval_prox` is retained for Argoverse 2,
SUNRGBD and Objectron. No GT-based scale alignment is applied.

## Prepare inputs

Download the public Stereo4D RGB split with the toolkit environment:

```bash
python -m scripts.prepare_wilddet3d_stereo --output data/stereo4d
```

For other datasets, obtain the annotations and RGB files through the
[official guide](https://github.com/allenai/WildDet3D/blob/main/docs/EVALUATION.md).
The public in-the-wild annotations are in
[`allenai/WildDet3D-Bench`](https://huggingface.co/datasets/allenai/WildDet3D-Bench),
not the training-only `WildDet3D-Data` repository named in older instructions.
ScanNet and Argoverse 2's prepared annotations and images are available in
[`RoyYang0714/3D-MOOD`](https://huggingface.co/datasets/RoyYang0714/3D-MOOD).
Use original RGB, not annotated or upsampled visualization images.

Export the oracle 2D inputs using the scorer environment:

```bash
$SCORER_PYTHON -m scripts.prepare_wilddet3d_perception \
  --upstream external/WildDet3D \
  --annotation data/stereo4d/annotations/Stereo4D_val.json \
  --data-root data/stereo4d --benchmark stereo4d \
  --output data/stereo4d_manifest.json
```

Other benchmark names are `in_the_wild`, `scannet`, `argoverse`, and
`omni3d_KITTI`, `omni3d_nuScenes`, `omni3d_SUNRGBD`, `omni3d_Hypersim`,
`omni3d_ARKitScenes`, `omni3d_Objectron`. Omni3D uses the official Omni3D-50
subset class mappings. Original images, including images with zero valid
prompts, remain in the manifest and evaluation coverage.

To explicitly evaluate an available-image subset, create a JSON list of the
excluded `file_path` strings and pass both `--exclude-image-paths missing.json`
and `--subset-annotation data/InTheWild_val_subset.json` when preparing inputs.
The adapter writes separate ground-truth annotations with those images and all
their annotations removed, preserving the original file. The manifest, metrics,
and summary record the evaluated/original image counts and exclusions. Use this
manifest for both inference and scoring. Subset results are not directly matched
to full-split paper scores; the official evaluator also recomputes category
frequency groups for the retained images.

For HDF5-backed ScanNet/Argoverse data, either install `h5py` in the encoder
environment or extract the original image byte arrays into `val/` under each
dataset root. Do not resize or re-encode them during extraction.

## Predict and score

The Small WDS518 checkpoint uses 518 for all three resolution arguments:

```bash
CUDA_VISIBLE_DEVICES=0 $ENCODER_PYTHON -m scripts.encode_wilddet3d_perception \
  --manifest data/stereo4d_manifest.json \
  --model-repo /path/to/spatial_encoder_v2_small \
  --checkpoint /path/to/checkpoint.pt --profile lite \
  --resolution 518 --spatial-resolution 518 --model-image-size 518 \
  --output results/perception/stereo4d/small

$SCORER_PYTHON -m scripts.score_wilddet3d_perception \
  --upstream external/WildDet3D --manifest data/stereo4d_manifest.json \
  --predictions results/perception/stereo4d/small \
  --output results/perception/stereo4d/small/metrics.json \
  --metrics dist bbox
```

For the existing Full e64 checkpoint, use `--profile full --resolution 1024
--spatial-resolution 504 --model-image-size 1008`. Resolution must match the
checkpoint; the adapter verifies every state-dictionary key and shape.
For the original Small e34 checkpoint, use the same three resolutions with
`--profile lite`.

Run multiple workers with the same output directory and distinct
`--shard-index 0..N-1 --num-shards N`. Each image is written atomically and
fingerprinted; identical runs resume. Scoring refuses missing images.

The model selects its highest predicted-IoU 3D candidate per oracle prompt.
There is no score threshold, GT 3D candidate selection, or NMS. Invalid geometry
is counted as rejected rather than removed from the ground truth. Prompts are
batched in groups of 32. The native normalized center/size is multiplied by
2.5 to obtain meters. Local XYZ sizes are converted to Vis4D's width/length/
height order as `[size_z, size_x, size_y]`; quaternions remain WXYZ.

`metrics.json` retains official raw scores on a 0–1 scale. Multiply AP and ODS
by 100 for paper-style tables. ATE is normalized by the official matching
threshold and is not an error in meters. `ODS`, `ODS_Sym`, and
`ODS_Canonical` are separate values; label the selected variant explicitly.
The paper specifies canonical rotations for mAOE in section 4.1, so its ODS
reference rows are paired with `ODS_Canonical` in the summary.

For the joint Omni3D score, keep all six `omni3d_<Subset>_manifest.json` files
under the run root and use the scorer environment:

```bash
$SCORER_PYTHON -m scripts.score_wilddet3d_omni_suite \
  --run-root results/perception --upstream external/WildDet3D \
  --data-root data/omni3d --models full_e64 small_e34 small_e100 --wait
```

It applies the official
joint category/image accumulation after all six subsets finish; it does not
average the six AP values.

## Background suite and result table

Configure `scripts.run_wilddet3d_perception` with `output`, `upstream`,
`encoder_python`, `scorer_python`, `gpus`, `num_shards`, `datasets`, and `models`.
Each dataset needs `name`, `manifest`, and optional `metrics`; each model needs
`name`, `model-repo`, `checkpoint`, `profile`, and the three resolution values.
Dataset manifests may arrive later: the queue waits for them before starting.
For concurrent suites sharing a result root, use distinct `status_file` names.
Set `max_existing_gpu_memory_mb` (for example, `500`) to wait for occupied GPUs
before assigning new inference jobs. A `comparison_config.json` in the result
root can specify ordered `models` (`name` and `label`), `status_files`, and
additional comparison `notes` so older and newer checkpoints appear together.
Start from [the three-model suite](../configs/perception_evaluation.example.json)
and copy [the comparison configuration](../configs/perception_comparison.example.json)
to `results/perception/comparison_config.json` to include all three columns.

```bash
nohup python -u -m scripts.run_wilddet3d_perception \
  --config configs/my_perception.json > logs/perception.log 2>&1 < /dev/null &
python -m scripts.summarize_wilddet3d_perception --run-root results/perception
```

The summary creates `comparison.md` and `comparison.csv`. Paper values are
labeled references. The paper's ScanNet/Argoverse table does not explicitly
identify prompt mode, so those rows should not be claimed as a matched-prompt
comparison without further confirmation. Report model training data and audit
cross-source image overlap before describing any score as zero-shot.
