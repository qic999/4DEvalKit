# Official evaluation code for the additional 4D benchmarks

**Runtime follow-up:** the [integration guide](additional_4d_evaluation.md) and
[executed-run report](../reports/additional_4d_20261002/comparison.md) supersede
the implementation-status column below. This document preserves the earlier
source audit; its "work needed" entries describe the state at that audit.

Checked **2026-10-02** against official repository file trees and scorer source
code. Of the 14 candidates, **10 have task-relevant scoring implementations**;
MLLM-4D also has a scorer, with the independent MLLM4D-Bench QA release still
unverified. HOI4D exposes a submission interface and preprocessing, MVVBench
publishes annotations without a scorer in its inspected release, and DA4D still
advertises its code as forthcoming.

**Official code available** describes the upstream release. **Integrated** in
the README describes a working interface in 4DEvalKit. The additional adapters
are not implemented by this audit. Existing boxes and answers must be converted
to each scorer's required inputs, and its ground truth and dependencies must be
available before a runtime evaluation is established.

## 4D perception

| Benchmark | Official code status and entry point | Required inputs / metric | Work needed in 4DEvalKit |
|---|---|---|---|
| **ADT** | **Available.** [`challenges/evaluate.py`](https://github.com/facebookresearch/projectaria_tools/blob/54fdff0b79caa94513b13265c6dd8e29ba664359/projects/AriaDigitalTwinDatasetTools/challenges/evaluate.py). | Prediction and annotation ZIPs containing timed object poses; GT vertices, symmetry definitions, diameters and object prototypes. Computes MSSD-based mAP across thresholds. | Export timestamped object-model poses and prototype IDs; resolve box-to-object-frame alignment. Obtain or construct protocol-compatible annotation archives. This pose score does not measure identity switches or velocity errors. |
| **HOI4D Object Tracking** | **Partial: no local scorer found in the official task repository.** [`ObjTracking`](https://github.com/hoi4d/ObjTracking/tree/afa7bf37b02b7e33cf245acd08b98c207ccc175c) contains only `README.md`, `process_data.py`, and `process_BundleTrack_data.py`. | README specifies `pred.npy` predictions for `test_wolabel.h5` and leaderboard submission. The two Python files prepare data. | Support the documented submission format where available, or implement explicitly labeled local pose/box-trajectory diagnostics using accessible GT. Do not call another HOI4D task's scorer the official object-tracking evaluator. |
| **HOT3D / HOT3D-Clips** | **Available via BOP.** Official [format converters](https://github.com/facebookresearch/hot3d/blob/146b34afef8c1a32adeef7e981c070109f225c87/hot3d/clips/bop_format_converters/README.md); [`eval_bop24_pose.py`](https://github.com/thodan/bop_toolkit/blob/cea62d651c7e395b2e1962b9749e4e89693c6ac4/scripts/eval_bop24_pose.py). BOP's dataset configuration explicitly supports HOT3D. | BOP pose CSV, object models, camera/GT files and targets. MSSD/MSPD-based pose scoring. | Export object IDs, rotation/translation and scores in the BOP convention; use the matching Aria/Quest split and object-frame alignment. Temporal tracking diagnostics require separate metrics. |
| **nuScenes Tracking** | **Available.** [`tracking/evaluate.py`](https://github.com/nutonomy/nuscenes-devkit/blob/b40adc467b919192899405d9b77871afee8efa07/python-sdk/nuscenes/eval/tracking/evaluate.py), `TrackingEval`. | Submission JSON with sample tokens, global-frame boxes, rotation, velocity, class, confidence and stable track IDs. AMOTA, AMOTP and tracking diagnostics. | Export full online tracks, including later entrants, in the official class/coordinate format. Local validation needs nuScenes metadata/GT. Hidden test results use the official submission service. |
| **TAPVid-3D** | **Available.** [`evaluate_model.py`](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid3d/evaluation/evaluate_model.py) and [`compute_tapvid3d_metrics`](https://github.com/google-deepmind/tapnet/blob/730cda1c730877cfedbe01bf87fb1cadb78a565d/tapnet/tapvid3d/evaluation/metrics.py). | Queried 3D point trajectories, visibility/occlusion, intrinsics, and declared scale-alignment mode. Average Jaccard, points within thresholds, occlusion accuracy. | Add a point-tracking output interface. An object box center or corner does not represent an arbitrary queried surface point. The scorer exists; the current model output is the limitation. |
| **DA4D / DetAny4D** | **Release unverified.** The [official project page](https://jarvishou829.github.io/DA4D/) still says `Code(Soon)`; no public evaluation entry point was located. | Paper describes sequence boxes, detection and temporal consistency measurements. | Keep conditional on a usable data/scorer release; do not present a paper-derived approximation as a reproduced official score. |

The previous survey did not identify ADT's challenge scorer. This audit corrects
that omission. The [official challenge instructions](https://eval.ai/web/challenges/challenge-page/2093/evaluation)
document a local example command. Challenge closure affects data/submission
access, not whether the scorer's source is public. ATEK additionally provides
[static 3D box evaluation](https://github.com/facebookresearch/ATEK/blob/aae40aa7d5b9573b5b81460467d7377c519127fb/atek/evaluation/static_object_detection/eval_obb3.py);
that is a separate protocol from ADT's object-pose challenge.

ADT and HOI4D evaluation preparation follows the existing
[split policy](adt_hoi4d_split_protocol.md): earlier ADT uses the same training
sequences per project-owner confirmation; HOI4D adopts the test-unseen project
assumption.

## 4D reasoning

| Benchmark | Official code status and entry point | Required inputs / metric | Work needed in 4DEvalKit |
|---|---|---|---|
| **MotionBench** | **Available.** [`compute_accuracy.py`](https://github.com/zai-org/MotionBench/blob/dcc9b0713c9b92d1c5d4ec7ee0b6dd51f86325a3/motionbench/metrics/compute_accuracy.py), called by `scripts/test_acc.py`. | Answer JSON keyed by QA UID and `video_info.meta.jsonl`; exact-choice overall/category accuracy. Public DEV has labels, TEST answers can be `NA`. | Convert predictions to UID/answer mapping and select labeled DEV consistently. Check full answer coverage and denominator handling before comparing scores. |
| **CLEVRER** | **Available inside the official baseline.** [`executor/run_mc.py`](https://github.com/chuangg/CLEVRER/blob/98b842082ba4f7c18b6b9e3f39145871782a65ef/executor/run_mc.py), [`run_oe.py`](https://github.com/chuangg/CLEVRER/blob/98b842082ba4f7c18b6b9e3f39145871782a65ef/executor/run_oe.py); `get_results.py` builds test submissions. | Validation annotations and symbolic-executor answers. Reports descriptive accuracy and per-option/per-question accuracy for explanatory, predictive and counterfactual tasks. | Reuse scoring semantics with LLM answers instead of the baseline executor. Each MC option is judged correct/wrong; a question is correct only when all its options are correct. `temporal_reasoning/eval.py` runs the dynamics model, not the final QA scorer. |
| **TempCompass** | **Available.** [`eval_multi-choice.py`](https://github.com/llyx97/TempCompass/blob/e1b463166400633e6061962d890a9ae85db29f70/eval_multi-choice.py), plus separate yes/no, matching and captioning scripts. | Predictions grouped by video and temporal aspect, answers and `meta_info.json`. MC supports `--disable_llm`; other answer formats have their own protocols. | Export predictions in the expected nesting, preserve aspect aggregation and choose the declared answer-extraction policy. MC can be scored without a hosted judge. |
| **4D-Bench** | **Available in inference examples.** [`qwen2_vl_7b_exp.py`](https://github.com/WenxuanZhu1103/4D-Bench/blob/f40f49a7539c4c5b1485ad5e80370cf006bbaf53/4D_Object_Question_Answering/eval_code_example/qwen2_vl_7b_exp.py) contains `extract_answer_option` and `handle_vqa_result`; captioning has a separate `eval_metrics` directory. | QA records with `Answer index`, multi-view clips and model answers. Example exports per-item correctness and parsed option. | Separate answer scoring from model inference and aggregate correctness with a fixed denominator, including invalid answers. Match the declared view/frame sampling; the inspected example uses views 1/8/16 and six frames per view. |
| **V-STaR** | **Available.** [`eval.py`](https://github.com/V-STaR-Bench/V-STaR/blob/1894682b15be0a0c0c5ab24be6e8252743cd6c2f/eval.py). | Answers, time intervals and timestamped 2D boxes with GT metadata. Qwen2.5-72B-Instruct judges semantic QA; the script computes temporal and spatial IoU and joint summaries. | Produce grounding outputs as well as text answers; provision the judge. Importing this script directly loads the 72B model, so isolate geometry-only diagnostics if needed. |
| **Physion V1.5** | **Available.** [`train_readout.py`](https://github.com/neuroailab/physion_evaluator/blob/03924d4a4c518044a279c10f286335bb11a1a709/physion_evaluator/train_readout.py), with the official feature-extraction interface. | Train/test feature HDF5 files, labels and scenario-index metadata; standardized classifier/readout accuracy on OCP/OCD tasks. | Implement the feature extractor and follow the readout training/test protocol. A boxes-to-LLM contact answer would be an explicitly different adaptation. |
| **MVVBench** | **No scorer found in the inspected official release.** [Pinned dataset files](https://huggingface.co/datasets/everex/MVVBench/tree/614331fd5a9f4543e5e59a36035d90e55c872b47) contain QA JSON/Parquet, `prepare_videos.py`, mapping CSV, README and attributes. | Public `correct_answer`, ordered views, category and static/dynamic labels enable local MC accuracy. | Add multi-view inputs and a declared exact-choice scorer with category/static/dynamic breakdown. This is feasible from the public labels, but a dataset preparation script is not evaluation code. |
| **MLLM4D-Bench** | **Scorer available; benchmark package only partly verified.** [`evaluation/evaluation.py`](https://github.com/GVCLab/MLLM-4D/blob/544fbd57627bbe789b92e42eb33ab4bf8e6a7fec/evaluation/evaluation.py) judges response JSON with Qwen3-VL-8B-Instruct. | Records with question, choices, answer and response. The checked examples use `real_mc` / `synthetic_mc`; the downloader also downloads VLM4D. | Confirm the independent MLLM4D-Bench question IDs and GT, then convert predictions and provision the judge. The [dataset root](https://huggingface.co/datasets/flow666/MLLM-4D-Datasets/tree/80c03a70c2d2cbe1088ee530ebfb4c512c7bd8f2) lists a test-video ZIP but no separately named test-QA file; ZIP contents were not inspected. |

## Integration implications

MotionBench DEV, TempCompass MC, CLEVRER validation and 4D-Bench QA have scoring
logic that can be reused or extracted once data/model-output adapters are added.
MVVBench can also be scored using its public answer labels after implementing
the multi-view adapter. These are implementation tasks, not reasons to wait for
a new upstream release.

ADT, HOT3D and nuScenes have official numerical scorers. Their adapter work is
primarily coordinate conventions, object identity, initialization and output
coverage. TAPVid-3D requires predictions for queried points. V-STaR needs explicit
grounding outputs and a judge, and Physion needs its prescribed readout protocol.

Source inspection also exposed details to check during integration: MotionBench
handles `NA` labels and answered-item counts differently, and mutates the
category field inside its QA loop; ADT's pose association should be checked on
multi-candidate fixtures before relying on it for multiple same-prototype
predictions. Preserve upstream behavior or document corrections with scorer
parity checks; do not silently alter the metric.

## Audit evidence and limits

The [audit manifest](../reports/upstream_eval_audit_20261002/manifest.json)
records exact GitHub commits, inspected file paths and SHA-256 hashes, and the
two Hugging Face dataset revisions/file inventories. Public source files were
read, including the metric calculations and command-line entry points.

This was a **source-level audit**. It did not download full benchmark datasets,
load judge models, execute the complete official evaluators, or generate new
benchmark results. Public code availability should therefore not be read as
runtime validation of every upstream environment.
