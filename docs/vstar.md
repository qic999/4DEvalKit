# V-STaR evaluation

The workflow supports predicted boxes, RGB, and RGB + boxes with one answerer.
It prepares all five released subtasks, resumes inference, and scores both
grounding chains plus semantic QA. Runtime validation is in progress; the
README remains **Not ready** until an actual model run and judging complete.

## Protocol

This implements the **conditioned subtasks** in the official
[inference example](https://github.com/V-STaR-Bench/V-STaR/blob/1894682b15be0a0c0c5ab24be6e8252743cd6c2f/inference_demo.py).
Some ground-truth information is deliberately provided by that protocol:

| Output | Supplied condition |
|---|---|
| Semantic answer | Original question |
| Temporal chain 1 | Released temporal question |
| Spatial chain 1 | Released spatial question and GT temporal interval |
| Spatial chain 2 | Released spatial question and GT box count |
| Temporal chain 2 | Released temporal question and GT boxes without timestamps |

These scores must not be described as end-to-end unconditioned grounding.
Neither these conditions nor the scoring labels initialize the spatial encoder
or select its frames. The semantic GT answer and reference reasoning chain are
never passed to the answerer. The judge sees the reference answer only after
inference. `tasks.json` contains declared task conditions; `scoring.json` holds
the separate scoring annotations.

The answerer receives 16 uniformly sampled frames by default, with timestamps,
and outputs boxes in original-image pixels. This frame budget and JSON output
instruction differ from the author's model-specific demo, which uses a video
input at 1 FPS. Record these differences when comparing with the paper.

The [pinned evaluator](https://github.com/V-STaR-Bench/V-STaR/blob/1894682b15be0a0c0c5ab24be6e8252743cd6c2f/eval.py)
uses **Qwen2.5-72B-Instruct**, ratings **0–3**, and correctness **rating >= 2**.
These code settings take precedence over the inconsistent rating description
in the upstream README. The toolkit executes the original judge function and
its generation defaults, with seed 0, and the unchanged numerical grounding
functions. It exports temporal recall/IoU, spatial AP/IoU, semantic accuracy,
AM/LGM, joint successes, domain breakdowns and duration breakdowns. Scores are
fractions except LGM. Perfect scores have an infinite LGM limit, represented by
`LGM: null, LGM_infinite: true` rather than invalid JSON.

Malformed grounding outputs count as zero with the full denominator. Failed
requests and truncated generations prevent scoring and can be resumed.

## Download and prepare

```bash
python -m scripts.download_vstar --output data/V-STaR
python -m scripts.download_vstar_judge --output data/models/Qwen2.5-72B-Instruct
python -m scripts.run_vstar prepare \
  --annotations data/V-STaR/V_STaR_test.json \
  --video-root data/V-STaR/videos --output data/prepared/V-STaR
```

The public video archive is 29.6 GB. The downloader verifies every annotated
video is present and checks archive CRCs during extraction. The archive and
extracted videos coexist. The 72B judge is an additional, separate download;
both downloaders reserve disk headroom and write `download_status.json`.
Add `--limit 2` to preparation for an explicitly labeled pilot.

## Inference and scoring

Generate proposals and predicted geometry from `manifest.json` with
`scripts.generate_media_proposals` and `scripts.encode_media_geometry` as in
the [temporal evaluation workflow](additional_4d_evaluation.md#temporal-qa).
Then, with a local OpenAI-compatible answerer server:

```bash
python -m scripts.run_vstar infer \
  --prepared data/prepared/V-STaR --mode rgb_boxes \
  --geometry results/vstar/geometry/geometry.json \
  --model qwen35_9b --base-urls http://127.0.0.1:8000/v1 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --output results/vstar/rgb_boxes

CUDA_VISIBLE_DEVICES=0,1,2,3 /path/to/judge/python -m scripts.run_vstar score \
  --prepared data/prepared/V-STaR \
  --predictions results/vstar/rgb_boxes/predictions.json \
  --judge-model data/models/Qwen2.5-72B-Instruct \
  --output results/vstar/rgb_boxes_metrics.json
```

The judge environment needs PyTorch, Transformers, Accelerate, NumPy and the
tokenizer dependencies. It must accommodate the full 72B model. The model
snapshot is pinned by `scripts.download_vstar_judge`; it is not replaced by a
smaller or quantized judge.

For the whole pipeline, copy
[the suite config](../configs/vstar.example.json) and fill in local paths:

```bash
nohup python -u -m scripts.run_vstar_suite \
  --config configs/local_vstar.json --output results/vstar_pilot \
  > logs/vstar_pilot.log 2>&1 < /dev/null &
```

The controller leases encoder, answerer and judge GPUs, stops its own servers
after use, and writes durable phase/status files. To run the full split, remove
`limit`, choose a new output directory, and set `pilot_status` to the completed
pilot's `status.json`. Without `judge_model`, the terminal state is explicitly
`inference_complete_scoring_pending`, not a completed evaluation.
