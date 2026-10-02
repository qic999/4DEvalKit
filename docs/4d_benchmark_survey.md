# Additional 4D benchmarks

Survey date: **2026-10-02**. These are candidates for integration, not supported
adapters or completed evaluations. Public release information was checked against
papers, author repositories, and dataset documentation; full dataset downloads
and end-to-end reproduction were not performed for this survey.

Follow-up source inspection found official scoring implementations for **10 of
the 14 candidates**, plus an MLLM-4D scorer whose independent benchmark QA release
remains unverified. See [official evaluation entry points](official_4d_evaluators.md)
for the code paths, input formats, and remaining adapter work. Upstream scoring
availability and integration into 4DEvalKit are tracked separately.

The priorities below are our assessment for evaluating SpatialEncoder's oriented
3D boxes and timestamped object tracks, optionally followed by a language model.
They are not rankings supplied by the benchmark authors.

## Current coverage and the missing measurements

The existing Stereo4D result evaluates **383 independent images**. It measures
geometry in dynamic scenes, but cannot establish tracking quality. STI-Bench,
VLM4D, and DSI-Bench evaluate answers about motion and changing spatial relations;
their QA accuracy does not measure the accuracy of the underlying trajectories.

Useful additions should cover complementary capabilities:

| Capability | Candidate datasets / benchmarks |
|---|---|
| Indoor object trajectories, camera motion, and occlusion | ADT; HOI4D; HOT3D |
| Outdoor 3D multi-object tracking and identity continuity | nuScenes Tracking |
| Temporal box stability in a static world | DA4D, subject to release availability |
| Arbitrary surface-point trajectories | TAPVid-3D, requiring additional model outputs |
| Metric object/camera motion reasoning | MLLM4D-Bench, subject to QA release verification |
| Fine-grained motion and temporal controls | MotionBench; TempCompass |
| Causal events and future physical outcomes | CLEVRER; Physion |
| Multiple synchronized views of dynamic scenes or objects | MVVBench; 4D-Bench |
| Joint question answering and temporal/spatial grounding | V-STaR |

## 4D perception candidates

“Derived protocol” means we would define and publish a box/trajectory evaluation
using the dataset's ground truth. It must be reported separately from an official
pose challenge or leaderboard.

| Candidate | What it adds | Release and scoring status | Fit and priority |
|---|---|---|---|
| **Aria Digital Twin (ADT)** | Real egocentric sequences with object poses, boxes, camera trajectories, and occlusions. | Public **MSSD-based mAP challenge scorer** in `projectaria_tools`; historical challenge is closed. Reuse it for matched pose inputs and available GT; additional box/trajectory diagnostics use a separately documented derived protocol. | **First batch**, using the [documented training sequence exclusions](adt_hoi4d_split_protocol.md). Align box and object-model frames, lens distortion, and timestamps. [Official scorer](https://github.com/facebookresearch/projectaria_tools/blob/54fdff0b79caa94513b13265c6dd8e29ba664359/projects/AriaDigitalTwinDatasetTools/challenges/evaluate.py), [challenge status](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/aria_digital_twin_dataset/adt_challenges). |
| **HOI4D Object Tracking** | Manipulated objects, rotation, occlusion, and category-level pose tracking. | Dataset and challenge preprocessing are public. The paper reports translation/rotation errors and 5-degree/5-cm success; the challenge repository documents prediction submission rather than a complete local scorer. | **First batch for rigid-object box trajectories**. Official pose comparison needs object-frame alignment and matched initialization; articulated parts require more outputs. [Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_HOI4D_A_4D_Egocentric_Dataset_for_Category-Level_Human-Object_Interaction_CVPR_2022_paper.pdf), [challenge code](https://github.com/hoi4d/ObjTracking). |
| **HOT3D / HOT3D-Clips** | Egocentric hand-object interactions with motion-capture object poses and scanned object models; synchronized views. | Public data/toolkit, BOP format converters, and **BOP 2024 pose scorer** with HOT3D dataset support. Associated hand-pose challenges have separate protocols. | **Second batch**, complementing ADT and HOI4D with additional capture conditions and object instances. Focus on rigid-object trajectories; derive GT boxes consistently from object models. A framewise BOP score alone is not a temporal tracking score. [Project and toolkit links](https://facebookresearch.github.io/hot3d/), [evaluation entry points](official_4d_evaluators.md#4d-perception). |
| **nuScenes Tracking** | Outdoor 3D multi-object tracking with stable identities and objects entering/leaving view. | Public devkit; validation can be scored locally. Official metrics include AMOTA, AMOTP, and identity switches. Hidden test labels require server evaluation. | **High scientific value; more integration work.** Export global-frame boxes, velocities, classes, confidence, and track IDs. Official tracking is online: no future frames. Full-scene coverage needs detection/association beyond first-frame slots. [Official protocol and scorer](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/tracking/README.md). |
| **TAPVid-3D** | Long-range tracking of arbitrary physical points, including visibility and occlusion. | Public data/code; evaluates metric 3D point trajectories with Jaccard-based scoring. | **Defer for the current box interface.** It requires tracking the queried surface point; box centers or corners are not interchangeable with those targets. [Official project](https://tapvid3d.github.io/). |
| **DA4D / DetAny4D** | Sequence-level 3D boxes and cross-frame geometric consistency. | Paper reports 3D AP and center/vertex temporal variation. The checked project page still labels code “Soon”; a runnable data/evaluation release was not verified. | **Watchlist.** Relevant to our output format, but its indoor consistency results should not stand in for moving-object tracking. [Project](https://jarvishou829.github.io/DA4D/), [paper and metric definitions](https://arxiv.org/html/2511.18814v1). |

Two protocol details materially affect comparison:

- HOI4D's paper initializes tracking with a perturbed GT object pose and evaluates
  RGB-D baselines. Our RGB plus initial 2D-box setup is a different setting.
  An oriented box's rotation also need not equal an object's semantic/CAD pose:
  origins, axis permutations, and symmetries must be resolved before pose scoring.
- DA4D's consistency calculation uses GT camera poses to move predictions into
  world coordinates. Variation around a fixed temporal mean is useful for a
  stationary object, but penalizes genuine motion. For moving objects, measure
  errors relative to the GT trajectory instead. This is a methodological
  inference from the published metric, not a proposed change to its official score.

### Proposed box-trajectory diagnostics

For ADT and the rigid-object portions of HOI4D/HOT3D, first establish a held-out
sequence manifest, an initialization rule, and a GT/prediction matching rule.
Then report these **derived** measurements alongside coverage:

| Measurement | Purpose |
|---|---|
| Mean / final observed-timestamp center error, in meters | Position accuracy through the observed sequence; these are tracking errors, not future forecasting scores. |
| Per-frame oriented 3D IoU and recall | Box geometry and missed observations. |
| Displacement / velocity error at declared time intervals | Motion accuracy after camera-motion compensation. |
| Track coverage, identity switches, fragmentation | Persistence through motion and occlusion; missed tracks must not disappear from the denominator. |
| Static-object jitter and moving-object residual error, separately | Distinguish instability from real motion. |
| Results by occlusion and track duration | Reveal failures hidden by a frame-weighted mean. |

For official nuScenes results, retain the original devkit's matching, filtering,
aggregation, and input-access rules. Keep single-camera or GT-prompted diagnostics
separate from a full benchmark submission. Existing Omni3D/nuScenes image AP is
not a tracking result.

## 4D reasoning candidates

All rows below evaluate semantic answers or grounding. They are not replacements
for numerical 3D perception metrics, even when the authors use the word
“perception” to describe a video QA task.

| Candidate | Added coverage | Public release / score | Integration assessment |
|---|---|---|---|
| **MotionBench** | Fine-grained video motion understanding across diverse scenes. | Official code and annotations; DEV answers are available, TEST answers use a leaderboard. Some source videos must be obtained separately and clipped using the provided mapping. | **First batch: DEV.** Fits a video QA adapter. Report motion categories separately; appearance and articulated motion need RGB/caption controls. [Official repository](https://github.com/zai-org/MotionBench). |
| **CLEVRER** | Descriptive, explanatory, predictive, and counterfactual questions about collision events. | Official train/validation videos, annotations, QA, code, and test evaluation entry point. | **First batch: validation.** Adds causal reasoning and prediction. Preserve its answer formats and question/option aggregation. Restrict inputs to the allowed observed video; never expose future target events. [Official project](https://clevrer.csail.mit.edu/). |
| **MLLM4D-Bench** | Six subtasks separating camera motion, object motion, and object-camera distance/direction; the paper reports 6k questions. | Public repository and a 10.4-GB test-video archive exist. The inspected root release does not separately list the benchmark QA; repository evaluation examples currently use VLM4D `real_mc` / `synthetic_mc`. Completeness of the independent QA release remains unverified. | **High priority once QA/split are verified.** Especially relevant to boxes plus camera trajectories. Do not count rerunning VLM4D as adding this benchmark. Match temporal anchors and the upstream answer-extraction/judge protocol. [Paper](https://arxiv.org/abs/2603.00515), [code](https://github.com/GVCLab/MLLM-4D), [released files](https://huggingface.co/datasets/flow666/MLLM-4D-Datasets/tree/main). |
| **MVVBench** | Real multi-view video reasoning: combining spatial and temporal evidence across cameras. | Public 1,323 QA over 608 multi-view samples, source-video mapping, and preparation script; **no scorer found in the inspected release**. Users obtain Ego-Exo4D, MMPTrack, and Panoptic source videos separately. | **Second batch.** Implement declared MC accuracy and synchronized multi-video loading, view labels, and cross-view identity handling. Report the provided static/dynamic breakdown alongside the full score. [Official dataset card](https://huggingface.co/datasets/everex/MVVBench). |
| **4D-Bench** | Multi-view understanding of animated 3D objects, with object QA and captioning. | Public dataset and task-specific evaluation code. | **Second batch: QA first.** Complements real-scene MVVBench. Whole-object boxes lose part articulation and appearance; use RGB and caption controls. Captioning adds separate metrics/judge requirements. [Official repository](https://github.com/WenxuanZhu1103/4D-Bench), [dataset](https://huggingface.co/datasets/vxuanz/4D-Bench). |
| **V-STaR** | Joint “what / when / where” video reasoning with temporal and spatial grounding. | Public data and evaluation scripts; combines QA accuracy, temporal IoU, and visual IoU. Official open-answer scoring uses Qwen2.5-72B-Instruct. | **Second batch.** Requires temporal intervals and framewise 2D grounding outputs, not only an answer letter. Report the components as well as the official aggregate. [Official repository](https://github.com/V-STaR-Bench/V-STaR). |
| **TempCompass** | Temporal understanding controls using conflicting video pairs and multiple answer formats. | Public processed videos, questions, and evaluation code, including an option without ChatGPT judging. | **Low-cost auxiliary addition: multiple-choice first.** Useful for checking temporal sensitivity, but not specifically metric 3D reasoning. Keep task formats distinct. [Official repository](https://github.com/llyx97/TempCompass). |
| **Physion V1.5** | Physical prediction, including future object contact. | Public download/evaluation code. Official workflow extracts features and trains a linear readout for OCP/OCD. | **Second batch / separate protocol.** A boxes-to-LLM yes/no experiment would be an adaptation, not a reproduction of the official probe result. Enforce each task's observation boundary. [Official evaluator](https://github.com/neuroailab/physion_evaluator). |

MLLM4D release inspection used GitHub revision
`544fbd57627bbe789b92e42eb33ab4bf8e6a7fec`. Test-video archive contents were not
downloaded or inspected. The default download helper also fetches training data
and VLM4D; an evaluation integration should select only its required files.

The existing COSMOS, EgoPlan-Bench2, and RoboVQA adapters can also broaden video
commonsense, planning, and interaction evaluation. They are already listed in
the [benchmark matrix](benchmark_matrix.md), so they should not be presented as
new adapters. Static-scene video QA and single-image visual-trace generation
remain separate from tests of moving-object trajectories.

## Integration order and comparison design

1. **Establish direct temporal perception.** Prepare held-out ADT and HOI4D rigid
   sequences under the [split protocol](adt_hoi4d_split_protocol.md), implement a
   common temporal manifest and derived trajectory scorer, and compare Full,
   Old Small, and WDS518 e100. Add HOT3D for complementary capture conditions.
2. **Add complementary QA first.** Integrate MotionBench DEV and CLEVRER
   validation; add TempCompass MC as a temporal control. Promote MLLM4D-Bench once
   its independent QA and scoring inputs are verified.
3. **Extend the temporal input pipeline.** Add online track initialization,
   association, and full-scene coverage for nuScenes; synchronized multi-view
   video input for MVVBench and 4D-Bench; grounding outputs for V-STaR.
4. **Keep specialized tasks explicit.** Physion needs its probe protocol;
   TAPVid-3D needs point trajectories; DA4D remains conditional on a usable release.

Before reporting results, resolve the following model-specific issues:

- **Training and evaluation splits:** the WDS518 e100 checkpoint's saved loader
  identity and manifest confirm train-only selection for 64 ADT and 522 HOI4D
  sequences. The project owner confirms that earlier ADT training used the same
  sequences. HOI4D evaluation adopts the project assumption that its test set
  was unseen during training. Use the [split protocol and sequence inventory](adt_hoi4d_split_protocol.md)
  to prepare held-out evaluation manifests. Report these as in-domain held-out
  evaluations; the dataset names also occur in training. Full and Old Small
  comparisons should record their own checkpoint-specific training inventories.
- **Input protocol:** distinguish predicted proposals, initial GT 2D prompts,
  repeated GT 2D prompts, and GT pose initialization. Supply GT identities only
  to the scorer, except for explicit target initialization allowed by a task.
- **Temporal coverage:** the current exporter initializes slots on the first
  sampled frame and does not stitch identities across clips. Short QA clips are
  not sufficient evidence of full-sequence tracking or recovery of later entrants.
- **Camera gauge and geometry:** transform all tracks into one declared frame;
  state whether camera poses are predicted or provided. Preserve metric scale.
  Separate a GT-camera diagnostic from an end-to-end predicted-camera result.
- **Observation boundary:** online tracking must be causal. Video QA can use the
  allowed clip, while prediction tasks must stop at the specified cutoff. Apply
  the same boundary to RGB, geometry, and captions.
- **Reasoning ablations:** retain boxes-only, caption-only, caption+boxes,
  RGB-only, and RGB+boxes under matched clips and reasoning settings. A separate
  GT-geometry arm can diagnose encoder versus reasoner errors where labels exist;
  report its extra information explicitly.
- **Temporal controls:** compare full tracks against a single-frame input and
  a time/order ablation. Use these as diagnostics, preserving the official
  full-input score. A question-only baseline helps detect language shortcuts.

The recommended expansion covers indoor manipulation, outdoor tracking,
occlusion, motion semantics, causal prediction, and multi-view temporal evidence.
Publish perception and QA scores in separate tables with their input protocols;
there is no meaningful shared average of meters, tracking accuracy, and QA accuracy.
