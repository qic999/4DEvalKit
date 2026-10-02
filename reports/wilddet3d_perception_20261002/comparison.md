# Perception evaluation

Inputs: single RGB image and GT 2D box prompts. Predictions: metric oriented 3D boxes. No LLM, GT depth, GT camera intrinsics or GT 3D boxes are supplied to the encoder.

Scoring uses the frozen WildDet3D evaluator (`1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a`). Scores below are on a 0–100 scale. Missing results are not zero scores.

| Dataset | Metric | Full e64 | Old Small e34 | Small WDS518 e100 | WildDet3D paper, no GT depth | State |
|---|---|---:|---:|---:|---:|---|
| stereo4d | AP (distance) | 7.30 | 5.31 | 4.96 | 7.5 | full_e64: scored; small_e34: scored; small_e100: scored |
| scannet | ODS (canonical) | 50.74 | 43.87 | 47.75 | 48.9 | full_e64: scored; small_e34: scored; small_e100: scored |
| argoverse | ODS (canonical) | 0.00 | 0.06 | 14.93 | 40.3 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_KITTI | AP (IoU) | 0.00 | 0.10 | 8.84 | 44.3 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_nuScenes | AP (IoU) | 0.00 | 0.01 | 9.04 | 35.3 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_SUNRGBD | AP (IoU) | 19.96 | 12.75 | 24.02 | 43.1 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_Hypersim | AP (IoU) | 10.70 | 8.68 | 10.51 | 17.3 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_ARKitScenes | AP (IoU) | 45.50 | 36.47 | 43.66 | 66.6 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_Objectron | AP (IoU) | 10.34 | 8.92 | 24.91 | 60.8 | full_e64: scored; small_e34: scored; small_e100: scored |
| omni3d_overall | AP (IoU) | 12.62 | 9.80 | 17.17 | 36.4 | full_e64: scored; small_e34: scored; small_e100: scored |
| in_the_wild (subset: 2460/2470 images) | AP (distance) | 6.65 | 7.72 | 11.63 | 24.8 | full_e64: scored; small_e34: scored; small_e100: scored |

## Reading the comparison

- **in_the_wild: 2460/2470 images.** 10 explicitly excluded images and all their annotations are removed from both inference and scoring. The paper value uses the full split and is not a matched-subset comparison. Category frequency groups are recomputed on this subset by the official evaluator; exclusions are recorded in [protocol.json](protocol.json).
- Old Small e34 is the original epoch-34 checkpoint used in the earlier reasoning evaluations. Its perception results are measured independently here.
- Native RGB / spatial / internal resolutions: Full e64 and Old Small e34 use 1024 / 504 / 1008; Small WDS518 e100 uses 518 / 518 / 518. All models use the same dataset manifests and scoring protocol.
- These are box-conditioned 3D regression scores, with GT category IDs attached for scoring; they do not measure open-vocabulary 2D category discovery.
- Table 5 in the paper does not explicitly label its prompt mode. Its ScanNet and Argoverse2 values are references, not a confirmed matched-prompt comparison.
- Model training distributions differ. The Small WDS518 e100 training configuration includes Omni3D and WildDet3D; do not describe all rows as zero-shot. Official evaluation splits are retained; cross-source training-image overlap has not been fully audited.
- Stereo4D uses 383 independent images from videos. It measures perception in dynamic scenes, not temporal tracking or trajectory consistency.
- ODS_Canonical is used above because paper section 4.1 specifies canonical rotations for mAOE. The evaluator also retains raw ODS/ODS_Sym in metrics.json. ATE here is normalized, not a translation error in meters.
- The old ScanNet val88 multi-frame AABB results use a different protocol and are excluded from this table.

Paper: https://arxiv.org/html/2604.08626v1 (Tables 3–6). Evaluator: https://github.com/allenai/WildDet3D.

Snapshot: 2026-10-02. All three models completed the same 53,341 images across ten datasets. The Omni3D aggregate reuses its six constituent datasets. [Raw metrics](metrics/) · [CSV](comparison.csv) · [Protocol](protocol.json).
