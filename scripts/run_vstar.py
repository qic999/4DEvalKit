"""Prepare, infer and score V-STaR conditioned QA and both grounding chains.

See docs/vstar.md. The model inference command never opens scoring annotations.
Native semantic judging loads Qwen2.5-72B-Instruct separately after inference.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from benchmark.vstar import PROTOCOL, TASKS, CONDITIONS, task_prompts, parse_prediction, summarize
from core.io import read_json, write_json, digest, source_signature
from core.runner import output_lock, read_journal


def prepare(a):
    rows = read_json(a.annotations)
    if a.limit is not None and a.limit < 1:
        raise ValueError('Limit must be positive')
    out = Path(a.output)
    videos = defaultdict(list)
    for path in Path(a.video_root).rglob('*.mp4'):
        videos[path.stem].append(path.resolve())
    manifest, tasks, scoring = [], [], []
    for i, row in enumerate(rows[:a.limit]):
        matches = videos[str(row['vid'])]
        if len(matches) != 1:
            raise ValueError(f"Video {row['vid']} has {len(matches)} matches; expected one")
        sid = f'vstar:{i}'
        manifest.append(dict(sample_id=sid, question=row['question'],
            media={'video': {'path': str(matches[0])}}, input_metadata={'scene': str(row['vid'])}))
        tasks.append(dict(sample_id=sid, prompts=task_prompts(row)))
        scoring.append(dict(sample_id=sid, annotation=row))
    if not manifest:
        raise ValueError('No questions selected')
    source = dict(annotations=source_signature(a.annotations), available=len(rows), selected=len(manifest),
                  limit=a.limit, selection='source_order', annotation_digest=digest(rows))
    public = dict(benchmark='V-STaR', protocol=PROTOCOL, source=source,
                  num_samples=len(manifest), conditions=CONDITIONS)
    write_json(out/'manifest.json', dict(public, samples=manifest))
    write_json(out/'tasks.json', dict(public, samples=tasks))
    write_json(out/'scoring.json', dict(public, samples=scoring))
    write_json(out/'status.json', dict(phase='prepared', **source))


def infer(a):
    from core.geometry import GeometryStore
    from core.inference import APIInferenceEngine
    from core.observations import visual_content, MATCHED_SYSTEM
    from core.prompts import make_messages
    prepared = Path(a.prepared)
    observations, tasks = read_json(prepared/'manifest.json'), read_json(prepared/'tasks.json')
    if tasks['protocol'] != PROTOCOL or observations['protocol'] != PROTOCOL:
        raise ValueError('Protocol mismatch')
    manifest = {r['sample_id']: r for r in observations['samples']}
    if len(manifest) != len(observations['samples']) or set(manifest) != {t['sample_id'] for t in tasks['samples']}:
        raise ValueError('Task and observation IDs must match uniquely')
    if a.mode != 'rgb' and not a.geometry:
        raise ValueError('Box modes require actual predicted geometry')
    geometry = GeometryStore(a.geometry) if a.mode != 'rgb' else None
    config = dict(protocol=PROTOCOL, tasks_digest=digest(tasks), manifest_digest=digest(observations),
                  media=[source_signature(r['media']['video']['path']) for r in manifest.values()],
                  mode=a.mode, model=a.model, endpoints=a.base_urls, video_frames=a.video_frames,
                  max_image_pixels=a.max_image_pixels, max_tokens=a.max_tokens, temperature=0, seed=0,
                  geometry=source_signature(a.geometry) if geometry else None,
                  extra_body=json.loads(a.extra_body), coordinate_output='original_image_pixels')
    engine = APIInferenceEngine(model=a.model, base_url=a.base_urls[0], base_urls=a.base_urls,
        max_tokens=a.max_tokens, timeout=600, extra_body=config['extra_body'])
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    with output_lock(out/'predictions.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != config:
            raise ValueError('Configuration changed; use a new output directory')
        write_json(out/'config.json', config)
        journal = out/'predictions.jsonl'
        previous = read_journal(journal)

        def complete(record):
            return (set(record.get('outputs', {})) == set(TASKS) and
                    all(x['status'] == 'ok' for x in record['outputs'].values()))

        def predict(task):
            sid = task['sample_id']
            if sid in previous and complete(previous[sid]):
                return previous[sid]
            # Decoded once for all five subtasks. Neither GT intervals nor boxes
            # are used to choose video frames or to initialize the encoder.
            visual = visual_content(manifest[sid], max_pixels=a.max_image_pixels,
                                    video_frames=a.video_frames) if a.mode != 'boxes' else []
            scene = geometry.lookup(sid)[1] if geometry else {}
            outputs = {}
            for name in TASKS:
                messages = make_messages(task['prompts'][name], scene, max_chars=600000, decimals=4)
                messages[0]['content'] = MATCHED_SYSTEM
                messages[1]['content'] = visual + [{'type': 'text', 'text': messages[1]['content']}]
                answer = engine.infer(messages)
                answer['prediction'] = parse_prediction(name, answer['raw_output']) if answer['status'] == 'ok' else None
                answer['parse_valid'] = answer['prediction'] is not None
                outputs[name] = answer
            return dict(sample_id=sid, outputs=outputs)

        selected = tasks['samples']
        with journal.open('a') as stream, ThreadPoolExecutor(a.workers) as workers:
            for record in workers.map(predict, selected):
                stream.write(json.dumps(record, allow_nan=False)+'\n'); stream.flush()
                previous[record['sample_id']] = record
                write_json(out/'status.json', dict(phase='inference',
                    completed=sum(complete(r) for r in previous.values()), total=len(selected)))
        results = [previous[t['sample_id']] for t in selected]
        failed = [r['sample_id'] for r in results if not complete(r)]
        write_json(out/'predictions.json', dict(protocol=PROTOCOL, config=config, results=results))
        write_json(out/'status.json', dict(phase='failed' if failed else 'inference_complete',
            completed=len(results)-len(failed), total=len(results), failed=failed,
            invalid_format=sum(not v['parse_valid'] for r in results for v in r['outputs'].values())))
        if failed:
            raise RuntimeError('Failed or truncated completions; resume after inspecting status.json')


def score(a):
    from benchmark.vstar import judge_namespace
    from core.official_4d import source, vstar_metrics
    scoring = read_json(Path(a.prepared)/'scoring.json')
    result = read_json(a.predictions)
    if scoring['protocol'] != PROTOCOL or result['protocol'] != PROTOCOL:
        raise ValueError('Protocol mismatch')
    if result['config']['tasks_digest'] != digest(read_json(Path(a.prepared)/'tasks.json')):
        raise ValueError('Predictions belong to different tasks')
    records = {r['sample_id']: r for r in result['results']}
    if len(records) != len(result['results']) or set(records) != {r['sample_id'] for r in scoring['samples']}:
        raise ValueError('Expected exactly one result per selected question')
    for r in records.values():
        if set(r['outputs']) != set(TASKS) or any(v['status'] != 'ok' or
            v.get('finish_reason') not in (None, 'stop', 'eos_token') for v in r['outputs'].values()):
            raise ValueError('Incomplete/truncated inference cannot be scored as complete')
    target = Path(a.output)
    rating_path = target.with_suffix('.ratings.json')
    identity = dict(predictions_digest=digest(result), scoring_digest=digest(scoring),
                    judge_model=source_signature(a.judge_model))
    with output_lock(target):
        ratings = read_json(rating_path) if rating_path.exists() else dict(identity=identity, scores={})
        if ratings['identity'] != identity:
            raise ValueError('Judge inputs changed; use a new output path')
        pending = [r for r in scoring['samples'] if r['sample_id'] not in ratings['scores']]
        if pending:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            config = read_json(Path(a.judge_model)/'config.json')
            # Reject a silent replacement of the official semantic judge.
            if config.get('model_type') != 'qwen2' or config.get('hidden_size') != 8192 or config.get('num_hidden_layers') != 80:
                raise ValueError('The official judge requires Qwen2.5-72B-Instruct weights')
            import torch
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_pretrained(a.judge_model, torch_dtype='auto', device_map='auto')
            tokenizer = AutoTokenizer.from_pretrained(a.judge_model)
            ns = judge_namespace(source('V-STaR-Bench/V-STaR', 'eval.py', a.cache), model, tokenizer)
            for row in pending:
                sid, gt = row['sample_id'], row['annotation']
                candidate = records[sid]['outputs']['answer_vqa']['prediction'] or ''
                rating = ns['qwen2_5_evaluation'](gt['question'], gt['answer'], candidate)
                ratings['scores'][sid] = rating if rating in (0, 1, 2, 3) else -1
                write_json(rating_path, ratings)
            del model
        metrics = vstar_metrics(a.cache)
        rows = []
        for row in scoring['samples']:
            sid, gt = row['sample_id'], row['annotation']
            pred = records[sid]['outputs']
            duration = gt['frame_count']/gt['fps']
            item = dict(sample_id=sid, domain=gt.get('domain', 'unknown'), rating=ratings['scores'][sid],
                        duration='Short' if duration < 60 else 'Medium' if duration < 180 else 'Long', chains={})
            for chain, suffix in [('1', ''), ('2', '_2')]:
                spatial = pred['answer_spatial'+suffix]['prediction'] or {}
                aps, miou = metrics['calculate_spatial_metrics'](gt['bboxes'], spatial)
                item['chains'][chain] = dict(spatial_iou=float(miou), spatial_ap=[float(x) for x in aps],
                    temporal_iou=metrics['calculate_temporal_iou'](gt['timestamps'], pred['answer_temporal'+suffix]['prediction']))
            rows.append(item)
        stats = dict(overall=summarize(rows))
        for field in ('domain', 'duration'):
            stats[field] = {k: summarize([r for r in rows if r[field] == k]) for k in sorted({r[field] for r in rows})}
        write_json(target, dict(protocol=PROTOCOL, conditions=CONDITIONS, source=scoring['source'],
            inference_config=result['config'], judge=identity, statistics=stats, results=rows,
            score_scale='0_to_1_except_LGM', note='Conditioned subtasks; not end-to-end grounding'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--annotations', required=True); q.add_argument('--video-root', required=True)
    q.add_argument('--limit', type=int); q.add_argument('--output', required=True)
    q = sub.add_parser('infer')
    q.add_argument('--prepared', required=True); q.add_argument('--geometry')
    q.add_argument('--mode', choices=['boxes', 'rgb', 'rgb_boxes'], required=True)
    q.add_argument('--model', required=True); q.add_argument('--base-urls', nargs='+', required=True)
    q.add_argument('--output', required=True); q.add_argument('--workers', type=int, default=4)
    q.add_argument('--video-frames', type=int, default=16)
    q.add_argument('--max-image-pixels', type=int, default=262144)
    q.add_argument('--max-tokens', type=int, default=8192)
    q.add_argument('--extra-body', default='{}')
    q = sub.add_parser('score')
    q.add_argument('--prepared', required=True); q.add_argument('--predictions', required=True)
    q.add_argument('--judge-model', required=True); q.add_argument('--output', required=True)
    q.add_argument('--cache')
    a = p.parse_args()
    globals()[a.command](a)


if __name__ == '__main__':
    main()
