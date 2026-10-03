# Additional 4D benchmark execution

Real checkpoint smoke runs completed on 2026-10-02. These small source-order / source-specific subsets validate execution; they are **not full benchmark results or paper comparisons**.

Model: **WDS518 epoch 100 Small**, input sizes 518/518/518. QA reasoner: **Qwen3.5-9B**, temperature 0, seed 0, thinking disabled, constrained final answers.

| Benchmark | Executed subset | Boxes | RGB | RGB + boxes |
|---|---|---:|---:|---:|
| MotionBench | 12 / 4018 answer records | 41.67 | 75.00 | 75.00 |
| TempCompass | 12 / 1580 answer records | 25.00 | 58.33 | 66.67 |
| 4D-Bench | 12 / 751 answer records | 25.00 | 33.33 | 41.67 |
| CLEVRER | 64 / 125852 answer records; 41 whole questions | 17.07 | 56.10 | 46.34 |
| MVVBench | 2 / 1323 answer records; one Panoptic sequence | 50.00 | 100.00 | 50.00 |

All 15 QA arms completed with `status=ok`, valid final-answer syntax and `finish_reason=stop`. Scores are percentages. CLEVRER uses per-question accuracy in this table; its per-option and four question-type breakdowns remain in the result files. MVVBench used full videos, without GT temporal crops.

Physion OCP also completed real geometry inference and the official logistic readout on **70 readout-training clips and 21 test clips**, three test clips per scenario. Its mean scenario accuracy was **33.33%**. This is a box-feature smoke experiment, not the full Physion score. Test labels were never used to select frames, normalize features or choose regularization.

## Full evaluations

Completed full-split results (same checkpoint/reasoner as above):

| Benchmark | Observation mode | Valid answers / full split | Score (%) |
|---|---|---:|---:|
| TempCompass | Boxes | 1,580 / 1,580 | 42.41 |

- **Running:** TempCompass (1,580 QA), 4D-Bench (751 QA), MotionBench labeled DEV (4,018 QA / 2,706 clips), all three observation modes. Seven GPU workers; complete-video sharding and prediction reuse across questions.
- **Queued after those runs:** CLEVRER validation, all 5,000 videos and 125,852 descriptive/candidate answer records, all three modes.
- **Physion full OCP running:** all 5,608 readout-training clips and 1,035 test clips are prepared. The complete box-feature/readout evaluation is running on GPU 7. Queue status: `results/additional_4d_20261002/physion_queue.json`.
- **MVVBench full split:** still needs the remaining EgoExo4D, MMPTRACK and Panoptic source videos. Only the explicit Panoptic smoke subset has been executed.

Local run roots: `results/additional_4d_20261002/{smoke,mvv_smoke,physion_smoke,full,clevrer_full,physion_full}`. Queue status: `results/additional_4d_20261002/clevrer_queue.json`. The full-run state is live in `status.json`; this report does not invent pending scores.

## Coverage of all 14 additions

| Benchmark | Verified state | Remaining requirement for a complete model result |
|---|---|---|
| MotionBench | Real model smoke complete; full DEV running | Finish queued GPU work |
| TempCompass | Real model smoke complete; full MC running | Finish queued GPU work |
| 4D-Bench | Real multiview model smoke complete; full QA running | Finish queued GPU work |
| CLEVRER | Real model smoke complete; full validation queued | Finish queued GPU work |
| MVVBench | Real two-view model smoke complete | Remaining source videos; no full-split result yet |
| Physion V1.5 | Real OCP box features + official readout complete on smoke subset | Full release run; OCD not implemented |
| ADT | Official numerical scorer executed on perfect/displaced pose fixtures | Held-out GT archive, object prototype IDs and object-frame pose alignment |
| HOI4D | Official preprocessing/submission interface inspected | Test data + submission-format predictor; no released local tracking scorer in inspected repo |
| HOT3D | Official evaluator launcher; current test clip actually inspected | Test object GT is absent; native object-model pose predictions and upstream environment |
| nuScenes Tracking | Pinned official evaluator launcher | Complete tracking dataset and global-coordinate online tracks, including later entrants |
| TAPVid-3D | Official numerical scorer executed on perfect/displaced point fixtures | Queried point trajectories and visibility; boxes cannot replace them |
| V-STaR | Official temporal/spatial functions executed on boundary fixtures | Actual grounded predictions and 72B semantic judge for the joint score |
| MLLM4D-Bench | Current test ZIP directory inspected | Independent test QA/GT; ZIP contains videos, no QA labels |
| DA4D / DetAny4D | Release audit | Public evaluation package remains unavailable |

## Verification

- The complete repository test suite passed: 123 tests, including decoder recovery and official Physion short-video clipping.
- Official MotionBench scoring exactly matches the real 12-item box-mode result.
- TempCompass matching and 4D-Bench parsing agree with upstream on tested edge cases.
- ADT, TAPVid-3D and V-STaR scorer fixtures are **synthetic metric checks**, not checkpoint evaluations.
- HOT3D `test_aria_bop_format/clip-003365.tar` is actually ZIP-format; its 453 entries contain images/camera JSON but no object GT. Dataset revision: `30fe9674782f32e1e5edba98476b6ff4300132c5`.
- The MLLM-4D test ZIP directory was inspected at dataset revision `80c03a70c2d2cbe1088ee530ebfb4c512c7bd8f2`; no independent QA annotations were present.

[Machine-readable evidence](runtime.json) · [Usage](../../docs/additional_4d_evaluation.md) · [Pinned upstream sources](../upstream_eval_audit_20261002/manifest.json)

MotionBench full-run decoding recovery: some released containers overstate their frame count. The reader now falls back to two-pass PyAV decoding and actual presentation timestamps, recording both declared and decoded frame counts. Three previously failing clips recovered all 16 requested observations; no questions were removed.

Physion full-run sampling recovery: the official MP4 loader clips OCP sample
indices to the last available frame. Preparation now checks decoded prefixes
and preserves this clipping, including repeated readout feature slots. Earlier
valid predictions remain reusable; short clips are retained in the evaluation.
