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

| Benchmark | Full split size | Boxes (%) | RGB (%) | RGB + boxes (%) |
|---|---:|---:|---:|---:|
| TempCompass | 1,580 | 42.41 | 69.87 | 70.38 |
| 4D-Bench | 751 | 34.62 | 66.31 | 66.05 |
| MotionBench labeled DEV | 4,018 | 41.66 | 61.72 | 61.72 |
| CLEVRER validation | 125,852 answer records | 14.61 | 50.09 | 43.55 |

Every reported QA score covers its complete split with valid, non-truncated
answers. CLEVRER reports overall **question** accuracy across 76,368 whole
questions; each multiple-choice question requires all its candidate decisions
to be correct. The other three benchmarks report answer accuracy.


CLEVRER full validation breakdown (question accuracy):

| Question type | Whole questions | Boxes (%) | RGB (%) | RGB + boxes (%) |
|---|---:|---:|---:|---:|
| Descriptive | 54,990 | 17.48 | 62.78 | 53.76 |
| Explanatory | 8,488 | 17.01 | 20.55 | 20.98 |
| Predictive | 3,557 | 0.00 | 30.98 | 34.24 |
| Counterfactual | 9,333 | 1.06 | 9.54 | 7.51 |

Use these question-type metrics for published CLEVRER comparisons. The
overall aggregate mixes question types and is not a replacement for them.
[CLEVRER full-result evidence](clevrer_full.json) also includes per-answer-record
metrics and source result signatures.

**Physion V1.5 OCP is complete:** 5,608 readout-training clips and 1,035 test
clips, with **58.64% mean scenario accuracy**. This run uses predicted box
features and the official logistic readout, separately from LLM reasoning.
The primary metric is the unweighted mean of the seven scenario accuracies,
not the accuracy weighted by the number of clips.

| Physion test scenario | Test clips | Accuracy (%) |
|---|---:|---:|
| Collision | 105 | 71.43 |
| Drop | 157 | 50.96 |
| Towers | 147 | 72.79 |
| Link | 157 | 54.14 |
| Roll | 157 | 45.22 |
| Contain | 153 | 56.21 |
| Dominoes | 159 | 59.75 |
| **Mean across scenarios** | **1,035 total** | **58.64** |

[Physion full-result evidence](physion_full.json) records feature coverage,
per-scenario counts, readout settings and the source result.

- **Complete:** TempCompass, 4D-Bench, MotionBench and CLEVRER, all three observation modes. Every answer passed coverage, syntax and non-truncation checks.
- **MotionBench:** all three modes have 4,018 / 4,018 valid answers. The release contains one single-option question; the parser accepts its public option without dropping the question or changing labels. All 4,018 geometry predictions are complete. Recovery status: `results/additional_4d_20261002/motionbench_recovery/status.json`.
- **CLEVRER complete:** all 5,000 validation videos, 125,852 descriptive/candidate answer records and 76,368 whole questions in each of the three modes. Geometry and QA coverage are complete.
- **Physion full OCP complete:** all 5,608 readout-training clips and 1,035 test clips have predicted geometry and exported features; the official readout has finished. Result: `results/additional_4d_20261002/physion_full/readout.json`.
- **MVVBench full split:** still needs the remaining EgoExo4D, MMPTRACK and Panoptic source videos. Only the explicit Panoptic smoke subset has been executed.

Local run roots: `results/additional_4d_20261002/{smoke,mvv_smoke,physion_smoke,full,motionbench_recovery,clevrer_full,physion_full}`. MotionBench recovery reuses its original geometry and QA journals under `full/MotionBench`; its separate controller status supersedes the interrupted original suite. MotionBench recovery and the downstream CLEVRER run have both completed. Queue status: `results/additional_4d_20261002/clevrer_queue.json`. This report does not invent pending scores.

## Coverage of all 14 additions

| Benchmark | Verified state | Remaining requirement for a complete model result |
|---|---|---|
| MotionBench | Full DEV complete in all three modes | None for this split and protocol |
| TempCompass | Full MC complete in all three modes | None for this split and protocol |
| 4D-Bench | Full multiview QA complete in all three modes | None for this split and protocol |
| CLEVRER | Full validation complete in all three modes | None for this split and protocol |
| MVVBench | Real two-view model smoke complete | Remaining source videos; no full-split result yet |
| Physion V1.5 | Full OCP box features + official readout complete | OCD not implemented |
| ADT | Official numerical scorer executed on perfect/displaced pose fixtures | Held-out GT archive, object prototype IDs and object-frame pose alignment |
| HOI4D | Official preprocessing/submission interface inspected | Test data + submission-format predictor; no released local tracking scorer in inspected repo |
| HOT3D | Official evaluator launcher; current test clip actually inspected | Test object GT is absent; native object-model pose predictions and upstream environment |
| nuScenes Tracking | Pinned official evaluator launcher | Complete tracking dataset and global-coordinate online tracks, including later entrants |
| TAPVid-3D | Official numerical scorer executed on perfect/displaced point fixtures | Queried point trajectories and visibility; boxes cannot replace them |
| V-STaR | Official temporal/spatial functions executed on boundary fixtures | Actual grounded predictions and 72B semantic judge for the joint score |
| MLLM4D-Bench | Current test ZIP directory inspected | Independent test QA/GT; ZIP contains videos, no QA labels |
| DA4D / DetAny4D | Release audit | Public evaluation package remains unavailable |

## Verification

- The complete repository test suite passed: 126 tests, including decoder recovery, official Physion short-video clipping and MotionBench single-option retention.
- Public answer constraints were checked on every prepared question: MotionBench 4,018; TempCompass 1,580; 4D-Bench 751; CLEVRER 125,852.
- Twelve full QA results passed exact sample coverage and final-answer audits. Physion feature arrays are finite and cover every train/test clip; scenario indices cover each split exactly once, and the reported mean matches the official readout output.
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

Repeated visual observations are now decoded once for adjacent questions. The
cache holds only one observation and includes media file signatures, ordered
views, sampling metadata and image settings. Question text and geometry remain
separate per sample. Cached visual payloads match the uncached payloads.
