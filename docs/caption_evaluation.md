# Caption and geometry comparisons

Run three matched text-input arms: caption only, caption + Full geometry, and
caption + Small geometry. Use the same frozen caption bundle in all arms.
Existing boxes-only results remain separate and can be included in the final
comparison when the model, geometry, questions and reasoning settings match.

## Run the background suite

Prepare the RGB manifests and geometry as described in [RGB evaluation](rgb_evaluation.md).
Copy [the caption config](../configs/caption_ablations.example.json) and edit
the paths and available GPU IDs. Each job requires published RGB media and
complete geometry for both variants.

```bash
mkdir -p logs
nohup python -u -m scripts.run_caption_ablations \
  --config configs/my_caption_ablations.json \
  --output-root results/caption_ablations \
  > logs/caption_ablations.log 2>&1 < /dev/null &
```

The controller loads one identical Qwen3.5-9B replica on each configured GPU.
Caption generation and QA both distribute requests across all replicas, including
when only one benchmark remains. It uses GPU leases and checks device occupancy.
Qwen3.5-9B is a multimodal model: the caption generator receives RGB, while the
QA calls receive only text. This is not a separate text-only model backbone.

## Input and answer controls

- Captions are generated from the same ordered images or at most 16 uniformly
  sampled video frames used by the RGB comparison. Images have at most 262,144
  pixels, JPEG quality 95, and videos retain real timestamps.
- The caption generator receives only pixels, view IDs, timestamps and a fixed
  description prompt. Questions, choices, answer labels, object proposals and
  predicted geometry are excluded by an explicit field allowlist. Video time
  intervals still come from the public input manifest, as in the RGB baseline.
- The prompt requests a factual description of at most 220 words, covering
  visible objects, appearance, spatial relations and changes across frames.
  The word count is a prompt instruction, not post-hoc trimming. Generation uses
  temperature 0, seed 0 and thinking disabled. It starts with 1,024 output tokens;
  failed or truncated captions retry with 2,048 and then 4,096. Attempts are saved.
  Only nonempty captions with normal completion are accepted.
- Captions are cached by media identity and generator settings. Images with
  identical file content can share a caption across questions; videos additionally
  retain their file identity and public time interval. Image order is significant.
- A full caption bundle must cover every manifest sample. Caption-only and
  caption + geometry share the same bundle digest, system prompt, questions,
  answer constraints and sampling settings. No RGB is decoded or passed to the
  reasoner during caption QA. The caption-only arm rejects geometry input.
- QA uses the existing native final-answer constraints and a 4,096-token limit.
  Missing or invalid responses fail the completion audit; samples are not dropped.

## Outputs and resume

`status.json` reports each GPU, caption preparation stage and QA arm. Each job
writes `captions.json` once all captions pass validation. Individual captions,
source identities, generation attempts and configuration digests are saved under
`caption_cache/`. Each QA arm writes its own `qa.json`, `.jsonl` journal and log.
`summary.json` and `summary.csv` update after each QA arm.

Rerun the same command with unchanged settings to resume. Valid captions and
responses are reused. Configuration changes require a new output root. Source
media changes invalidate their caption cache identities. Do not replace captions
inside an existing QA run.

For an individual arm with a completed caption bundle:

```bash
python eval.py --benchmark Q-Spatial-Bench \
  --data /path/to/QSpatial_plus.parquet --split QSpatial_plus \
  --media-manifest data/prepared/qspatial/QSpatial_plus/manifest.json \
  --captions results/caption_ablations/qspatial_plus/captions.json \
  --observation-mode caption --answer-format native \
  --model qwen35_9b --base-url http://localhost:8000/v1 \
  --temperature 0 --seed 0 --max-tokens 4096 \
  --extra-body '{"chat_template_kwargs":{"enable_thinking":false}}' \
  --output results/caption_only.json --resume
```

Use `--observation-mode caption_boxes --geometry /path/to/geometry.json` for
caption + boxes. To distribute a single evaluation across identical replicas,
replace `--base-url` with `--base-urls URL1 URL2 ...` and adjust `--concurrency`
and `--batch-size`. Replica URLs are recorded in the run configuration.
