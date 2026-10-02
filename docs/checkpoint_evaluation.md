# Evaluate another encoder checkpoint

Use `scripts.run_checkpoint_evaluation` to regenerate geometry for an existing
benchmark suite and evaluate `boxes`, `rgb_boxes`, and `caption_boxes` with a
fixed reasoner. It reuses prepared media, first-frame 2D proposals, and complete
question-independent caption bundles. It does not reuse old 3D predictions.

## Configure the run

Start from [the example configuration](../configs/checkpoint_evaluation.example.json).
Set the model repository, checkpoint, encoder Python environment, local reasoner
model, GPUs, and prepared input paths. The toolkit environment needs its normal
dependencies; the encoder environment needs the model repository's dependencies.
The reasoner environment must provide vLLM with the model's multimodal support.

Use the input sizes required by the checkpoint. For
`se-small-lite-wds518-8xh100-e100`, set `resolution`, `spatial_resolution`, and
`model_image_size` to `518`. The historical defaults are `1024`, `504`, and
`1008`, respectively. A comparison across these configurations changes both
weights and encoder resolution.

Each job uses the same benchmark name, split, media manifest, and proposal
directory as the earlier evaluation. Add `captions` only when the bundle covers
the entire manifest. Missing caption bundles are marked `unavailable`; other
observation modes still run. RGB-only and caption-only baselines can be reused
when their inputs and reasoning settings are identical.

The model repository must expose the native evaluation interface used by
`scripts.encode_media_geometry`, including open-vocabulary object slots and
predicted camera poses. Pin its revision and record any evaluation adapters.
The media encoder verifies that every checkpoint key and tensor shape matches
the instantiated model.

## Validate and launch

```bash
python -m scripts.run_checkpoint_evaluation \
  --config configs/my_checkpoint.json \
  --output-root results/my_checkpoint --validate-only

mkdir -p logs
nohup python -u -m scripts.run_checkpoint_evaluation \
  --config configs/my_checkpoint.json \
  --output-root results/my_checkpoint \
  > logs/my_checkpoint.log 2>&1 < /dev/null &
```

Validation checks proposal coverage, manifest IDs, and caption identity. GPU
workers acquire toolkit leases and wait for available memory. All configured
GPUs first encode disjoint shards, then host identical reasoning replicas.
Completed encoder samples and QA responses are resumable with the same command
and unchanged inputs. Stop the previous controller before resuming.

The reasoning protocol uses temperature 0, seed 0, native final-answer
constraints, a 4,096-token output budget, 16 sampled video frames, and the same
observation prompt across modes. The launcher checks exact sample coverage,
valid final answers, and non-truncated completion reasons before publishing a
score. Bounded `--limit` encoder smoke checks belong in a separate output
directory and are not full-benchmark results.

## Optional ScanNet evaluation

Add a `scannet` object with `name`, `benchmark`, `split`, `dataset`, `data`,
`manifest`, and `data_root`. The model repository must contain
`inference_gt2d/scannet_val88.txt`, `scene_inference.py`, `merge_json.py`,
`make_balanced_shards.py`, and `compute_metrics_3d.py`.

This phase evaluates all 88 scenes using GT 2D proposals/categories and GT
camera poses, merges predictions, computes direct 3D metrics, and runs the
ScanNet QA subset in `boxes` mode. It is separate from the predicted-proposal,
predicted-camera protocol used for the image/video benchmark suite.

## Outputs

- `status.json`: controller phase and per-shard/per-arm state.
- `process.json`, `config.json`, `input_digests.json`: process identity and frozen inputs.
- `media/<job>/geometry.json`: complete validated geometry after shard merging.
- `scannet/metrics.log`: direct ScanNet 3D metrics when enabled.
- `qa/<job>/<variant>/<mode>/qa.json`: answers and native scores.
- `summary.json` / `summary.csv`: completed, failed, and unavailable arms.

An optional `baseline_table` can reference the CSV produced by
`scripts.consolidate_conversation_results`. Set each job's `table_name` to the
matching `Benchmark / split` row. The launcher preserves those rows and adds
new-checkpoint scores in `comparison_all_results.csv`, an Excel workbook when
`openpyxl` is installed, and `comparison.md`. The current comparison column
labels identify the WDS518 epoch-100 experiment. Paper values remain reference
scores with potentially different dataset splits and evaluation protocols.
