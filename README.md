# 4DEvalKit

Run benchmark evaluation with scene geometry and a text LLM, or score saved predictions offline.

## Install

Use Python 3.10 or newer. Run commands from the repository root.

```bash
git clone https://github.com/qic999/4DEvalKit.git
cd 4DEvalKit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python eval.py --list-benchmarks
python eval.py --help
```

For live inference, provide an OpenAI-compatible `/v1/chat/completions` endpoint. Use `--model` for the model name served by that endpoint and `--base-url` for its URL ending in `/v1`. If authentication is required, set an environment variable and select it with `--api-key-env`.

## Prepare evaluation inputs

Each evaluation needs benchmark questions and either a geometry file or saved model responses.

- `--data`: a local dataset path or Hugging Face dataset ID. Omit it to use the benchmark's configured source.
- `--geometry`: scene objects or timestamped object tracks, keyed by sample ID or a supported source ID. VSI-Bench also accepts scene names.
- `--predictions`: saved responses for offline scoring.

Export a sample manifest to obtain the IDs and media references needed to prepare geometry:

```bash
python eval_mmsi_bench.py --data /path/to/MMSI-Bench \
  --export-manifest data/manifests/mmsi.json
```

Add `--export-media-dir data/media/mmsi` to export embedded images or frames. Generate geometry for the same dataset version, split, and sample order. See [Geometry format](docs/geometry_schema.md) for object, track, timestamp, and camera-pose fields.

Convert existing scene boxes when needed:

```bash
python -m scripts.convert_geometry --format legacy \
  --input /path/to/scene_boxes.json \
  --output data/geometry/vsi_bench.json \
  --units m --coordinate-frame world_z_up \
  --geometry-source predicted --encoder my_encoder \
  --checkpoint /path/to/checkpoint.pt \
  --proposal-source predicted_2d --label-source detector \
  --camera-pose-source predicted
```

Set provenance fields to match how the inputs were produced. For merged 10D boxes, use `--format merged --quaternion-order xyzw`; for a directory of scene files, also provide `--merged-name filename.json`.

## Evaluate a benchmark

Validate the question-to-geometry mapping before sending requests:

```bash
python eval_vsi_bench.py \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --geometry data/geometry/vsi_bench.json \
  --dry-run --output results/vsi_check.json
```

Run inference and scoring:

```bash
python eval_vsi_bench.py \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --geometry data/geometry/vsi_bench.json \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 0 --max-tokens 512 \
  --concurrency 8 --batch-size 16 \
  --run-label my_encoder \
  --output results/vsi_scannet.json --resume
```

The unified entry point is equivalent:

```bash
python eval.py --benchmark VSI-Bench \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --geometry data/geometry/vsi_bench.json \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --output results/vsi_scannet.json --resume
```

Use `--save-prompts` to retain prompts and `--limit 10` for a small trial. Pass endpoint-specific generation options with `--extra-body`, for example:

```bash
--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

## Evaluate dynamic scenes

Provide `tracks[].observations[]` with stable track IDs, timestamps in seconds, and per-frame boxes in a consistent coordinate frame. Use `--require-tracks` to validate that tracks are present.

```bash
python eval_sti_bench.py --data /path/to/STI-Bench/qa.parquet \
  --geometry data/geometry/sti_bench.json --require-tracks \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --output results/sti_bench.json --resume

python eval_vlm4d.py --split real_mc \
  --geometry data/geometry/vlm4d_real.json --require-tracks \
  --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1 \
  --output results/vlm4d_real.json --resume
```

Run VLM4D's `real_mc` and `synthetic_mc` splits separately. DSI-Bench accepts `--split std` or `--split all`; supply matching geometry for each video augmentation. See [Scoring protocols](docs/metric_protocols.md) for metric definitions.

## Score saved predictions

Offline scoring does not require a running model endpoint:

```bash
python eval_vsi_bench.py \
  --data /path/to/vsibench_qa.json --dataset scannet \
  --predictions /path/to/saved_responses.json \
  --model YOUR_MODEL --output results/vsi_replay.json
```

VSI-Bench defaults to `--vsi-metric-protocol physbrain`. Select `--vsi-metric-protocol project` to use the legacy project scorer. Use the same protocol when comparing runs.

## Run multiple benchmarks and models

Copy the suite template and edit the endpoint, dataset paths, benchmarks, and geometry paths for each model:

```bash
cp configs/core_suite.example.json configs/local_core_suite.json
```

Paths in the configuration are relative to the repository root. Keep shared inference settings under `defaults`; put each model's geometry paths under `variants`.

```bash
# Preview commands.
python -m scripts.run_suite --config configs/local_core_suite.json --mode plan

# Validate inputs without calling the LLM.
python -m scripts.run_suite --config configs/local_core_suite.json \
  --mode check --output-dir results/preflight

# Run in the background.
bash scripts/run_background.sh logs/core_suite.log \
  .venv/bin/python -u -m scripts.run_suite \
  --config configs/local_core_suite.json \
  --mode run --output-dir results/core_suite --resume

tail -f logs/core_suite.log
```

The background launcher records a PID and redirects output to the selected log. Each suite job has its own result and log; `suite_index.json` records job exit codes.

## Resume and inspect results

Rerun the same command with `--resume` to reuse successful responses and retry incomplete requests. Keep the dataset, geometry, model, and inference settings unchanged when resuming. Use a new output path for a changed experiment or another dry run.

Each evaluation writes:

- `.json`: final scores, response records, and status counts.
- `.jsonl`: responses saved incrementally during execution.
- `.manifest.json`: configuration and input fingerprints used for resuming.

Create a comparison CSV:

```bash
python -m scripts.summarize_results results/core_suite \
  --output results/core_scores.csv
```

## Try the included examples

These examples use synthetic fixtures and do not require model weights:

```bash
python eval_vlm4d.py --data examples/dynamic_qa.json \
  --geometry examples/dynamic_geometry.json --require-tracks \
  --dry-run --output results/example_check.json

python eval_vlm4d.py --data examples/dynamic_qa.json \
  --predictions examples/dynamic_predictions.json \
  --model synthetic_fixture --output results/example_replay.json
```

To run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

[Geometry format](docs/geometry_schema.md) · [Scoring protocols](docs/metric_protocols.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)
