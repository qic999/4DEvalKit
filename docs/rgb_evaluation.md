# Evaluate Full/Small from RGB media

Prepare the benchmark manifests first using `scripts.prepare_benchmarks`.
The RGB pipeline uses the native SpatialEncoder repository and checkpoints,
GroundingDINO for proposals, and a local Qwen reasoning server.

## Environments

Use the toolkit environment to run the controller. Set `encoder_python` to an
environment that can import the model repository, SAM3 and Depth Anything 3.
Set `llm_python` to an environment with Transformers, GroundingDINO, torchvision,
OpenCV and vLLM. `detector_model` and `llm_model` are downloaded local checkpoint
directories. All workers run from the toolkit root.

Copy [the example config](../configs/media_models.example.json) and fill in paths.
Give each dataset split a unique job name. `manifest` is the exported media
manifest; `data` is the corresponding native annotation source used for scoring.

```bash
mkdir -p logs
nohup python -u -m scripts.run_media_models \
  --config configs/my_media_models.json \
  --output-root results/media_models \
  > logs/media_models.log 2>&1 < /dev/null &
```

The example reserves GPU 0 for detection, GPUs 1–3 for Full, GPUs 4–6 for Small,
and GPU 7 for reasoning. Choose unused devices before launching. Optional
`wait_for_scannet_status` points to a ScanNet recovery `final_status.json`; the
reasoner waits for that controller to release its GPU. Omit the field for an
independent run.

## Inputs and geometry

The `fixed_vocab_first_frame_v1` protocol uses the same detections for both
encoders. Its settings are recorded with every result:

- The fixed vocabulary is in `configs/detector_vocabulary.json`. GroundingDINO
  receives RGB images and vocabulary text. Questions and answer annotations
  are excluded from detection and encoder inputs.
- Detection thresholds are 0.25 for boxes and 0.20 for text, with class-agnostic
  NMS at IoU 0.65 and a configurable object cap (default 40).
- Images retain their source order. Videos use up to 16 uniformly sampled
  frames within the published question interval, or the full video when no
  interval is supplied. Timestamps remain relative to the original video.
  Single-time annotations select the nearest frame; an annotation exactly at
  the container duration selects the last frame with its actual timestamp.
- The first sampled frame initializes object slots. Later frames propagate the
  native tracker. Every sampled frame has an inference stage; later stages do
  not receive conditioning boxes. Objects entering later can be missed.
- Native boxes are converted to meters using the training normalization factor
  2.5. Native box quaternions are converted from wxyz to xyzw. Dynamic geometry
  uses predicted camera poses and a first-camera coordinate frame. Multi-image
  geometry retains view-local boxes and predicted camera transforms.
- Samples without detections remain in the evaluation denominator. Per-variant
  coverage reports include the number of empty scenes.
- The text reasoner uses a 131,072-token context. Geometry in its prompt is
  rounded to four decimal places with `--geometry-decimals 4`; full precision
  remains in the saved geometry. QA results record this formatting setting.

This protocol evaluates a detector–encoder–reasoner pipeline. Record it
separately from the ScanNet GT-2D/GT-category/GT-camera-pose setting. Compare Full
and Small using the same vocabulary, object cap, frames and reasoning settings.
The first-frame proposal strategy and fixed vocabulary limit coverage of novel
objects, later entrants, appearance attributes and semantic facing directions.

## Outputs and resume

`status.json` records proposal, geometry and QA progress independently. Each job
contains:

- `proposals/index.json` and `proposals/samples/`: shared detector outputs.
- `<variant>/shard_*/samples/`: durable per-question geometry and diagnostics.
- `<variant>/geometry.json`: validated geometry merged from all shards.
- `<variant>/qa.json` and `qa.jsonl`: scores and resumable model responses.
- `logs/`: detector, encoder-shard and QA logs.

Rerun the same command to resume. Keep configurations and checkpoints fixed.
Missing or duplicate question IDs fail the geometry merge. Request failures
fail the QA job; truncated model responses remain recorded as failures in the
benchmark denominator. Aggregate scores appear only after all questions in the
split have been covered.

To recover only failed jobs, pass their configured names with `--jobs`:

```bash
nohup python -u -m scripts.run_media_models \
  --config configs/my_media_models.json \
  --output-root results/media_models \
  --jobs sti_bench_train dsi_bench_all \
  > logs/media_models_recovery.log 2>&1 < /dev/null &
```

Unselected jobs keep their previous status and results. Video open/decode
failures are retried up to three times with a fresh decoder; invalid time
intervals still fail immediately. Cached geometry must match the original
input and checkpoint fingerprints before it can be reused.

## Longer answers without rerunning the encoders

Inspect `finish_reason` and `usage.completion_tokens` in `qa.json`. A response
that ends with `length` at the configured output limit is truncated. The default
limit is 512 tokens. In a copy of the run config, set `qa_max_tokens` to 4096 and
`qa_timeout` to 300, then run a separate QA evaluation from the saved geometry:

```bash
nohup python -u -m scripts.run_media_models \
  --config configs/my_long_answer_run.json \
  --geometry-root results/media_models \
  --output-root results/media_qa4096 \
  --jobs qspatial_plus \
  > logs/media_qa4096.log 2>&1 < /dev/null &
```

This checks complete sample coverage and evaluates every question for both
variants with the new output budget. It preserves the original geometry and
scores, and records the new token limit in the QA manifest. Keep the prompts,
models and scoring settings fixed when comparing budgets. Longer budgets can
still truncate unusually verbose answers; check the resulting status counts.

When another controller still needs the reasoning GPU, add
`"wait_for_controller": {"pid": 12345, "start_time": "..."}` to the new config.
Use `scripts.finish_scannet_recovery.process_identity(pid)` for the recorded
start time. The new run waits for that exact process to exit before loading its
reasoning server. Do not run two independent controllers on the same GPU
without coordinating their lifetimes.

For individual stages, run `python -m scripts.generate_media_proposals --help`
and `python -m scripts.encode_media_geometry --help` in the corresponding
environments, then pass the resulting geometry to the normal `eval.py` CLI.

## Native answer formats and matched observation ablations

Increasing the output budget alone may still produce truncated explanations.
Use `--answer-format native` with a vLLM server supporting
[structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
The media controller exposes this as `"qa_answer_format": "native"` in its
configuration. It constrains choice tasks to their public option letters, SAT
to the displayed answer texts, and Q-Spatial to its scalar-and-unit format.
VSI numeric responses use decimal syntax. Numeric fields allow at most 12 integer
digits and eight fractional digits; Q-Spatial accepts m, cm, mm, ft or in.
These constraints use no answer labels. Results record `native_answer_v1` and
the per-question constraint. Keep these scores separate from free-form runs.

Matched ablations use a common system prompt and the following observation modes:

| CLI mode | Input |
|---|---|
| `--observation-mode rgb` | RGB only; omit `--geometry` |
| `--observation-mode boxes` | Geometry only |
| `--observation-mode rgb_boxes` | The same RGB inputs plus geometry |

All modes require `--media-manifest`. Visual arms keep image ordering and use the
same 16-frame video sampling as the encoder bridge, with real timestamps. Images
are resized to at most 262,144 pixels before JPEG encoding at quality 95. The
multimodal server uses matching pixel limits and supports up to 32 images.
Answers use the same model, prompt, sampling settings and native constraints in
every arm. RGB-only is shared between Full and Small; the other modes run once
for each encoder. Complete datasets are scored, including empty detections.

For an unattended sequence, configure
[`observation_ablations.example.json`](../configs/observation_ablations.example.json):

```bash
nohup python -u -m scripts.run_observation_ablations \
  --config configs/my_observation_ablations.json \
  --output-root results/observation_ablations \
  > logs/observation_ablations.log 2>&1 < /dev/null &
```

The sequence waits for the recorded native-answer run, checks complete sample
coverage and verifies every response ended normally and matches its declared
format. Any truncation or missing answer prevents comparisons from starting.
It then waits for the source evaluation controller to finish, loads one
multimodal reasoner per available configured GPU, and runs the comparisons.
Controllers record process start times to distinguish restarted or reused PIDs.
Reasoning servers coordinate through GPU leases and check current GPU occupancy.
Per-task progress and scores are saved in `status.json`; each task has its own
`qa.json`, journal and log. Interrupted tasks can resume with the same config.

For a ScanNet oracle, `scripts.prepare_scannet_oracle` exports GT object boxes
into the native encoder's world frame, using the dataset's translations and
checking against camera-frame annotations. It excludes aggregate room size,
object counts and question answers. Use the exported manifest and geometry in
a separate `boxes`-only job, together with Full and Small geometry. This oracle
has all annotated objects and GT categories, so it measures a broader upper
reference than replacing only predicted box coordinates.
