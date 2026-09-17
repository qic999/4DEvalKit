"""Generate shared question-independent captions, then run text-only caption/box QA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import datetime
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from core.captions import VERSION, SYSTEM, PROMPT, CaptionStore, caption_messages, media_identity, public_media
from core.inference import APIInferenceEngine
from core.io import digest, read_json, write_json
from core.response_constraints import matches_constraint
from core.runner import output_lock
from scripts.finish_scannet_recovery import process_identity
from scripts.reasoner_pool import ReasonerPool

ROOT = Path(__file__).resolve().parents[1]


def caption_config(config):
    return {'protocol': VERSION, 'model': config['llm_name'], 'model_path': config['llm_model'],
            'system': SYSTEM, 'prompt': PROMPT, 'temperature': 0, 'seed': 0,
            'max_tokens_attempts': config.get('caption_token_budgets', [1024, 2048, 4096]),
            'max_image_pixels': config.get('max_image_pixels', 262144), 'video_frames': 16,
            'jpeg_quality': 95, 'question_conditioned': False, 'geometry_conditioned': False,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}}}


def generate_caption(row, identity, key, config, engines):
    messages = caption_messages(row, max_pixels=config['max_image_pixels'], video_frames=16)
    attempts = []
    for budget, engine in zip(config['max_tokens_attempts'], engines):
        response = engine.infer(messages)
        attempts.append({'max_tokens': budget, **response})
        if response['status'] == 'ok' and response.get('finish_reason') in {'stop', 'eos_token'}:
            return {'caption_key': key, 'config_digest': digest(config), 'media_identity': identity,
                    'caption': response['raw_output'], 'status': 'ok', 'finish_reason': response['finish_reason'],
                    'attempts': attempts}
    return {'caption_key': key, 'config_digest': digest(config), 'media_identity': identity,
            'caption': '', 'status': 'failed', 'finish_reason': attempts[-1].get('finish_reason'),
            'attempts': attempts}


def prepare_captions(job, config, output, endpoints, update):
    manifest = read_json(job['manifest']); cfg = caption_config(config)
    samples = manifest['samples']; unique = {}; sample_keys = {}; files = {}
    destination = output/job['name']/'captions.json'
    update(phase='indexing_media', samples=len(samples))
    for index, row in enumerate(samples):
        if row['sample_id'] in sample_keys:
            raise ValueError('Duplicate public sample IDs')
        identity = media_identity(row, files)
        key = digest({'media_identity': identity, 'config': cfg})
        unique.setdefault(key, (public_media(row), identity)); sample_keys[row['sample_id']] = key
        if index % 100 == 0:
            update(phase='indexing_media', indexed=index+1, unique_media=len(unique))
    cache = output/'caption_cache'; cache.mkdir(parents=True, exist_ok=True)
    records = {}; pending = []
    for key, (row, identity) in unique.items():
        path = cache/f'{key}.json'
        record = read_json(path) if path.exists() else None
        if (record and record.get('status') == 'ok' and record.get('finish_reason') in {'stop', 'eos_token'}
                and record.get('caption') and record.get('caption_key') == key
                and record.get('config_digest') == digest(cfg) and record.get('media_identity') == identity):
            records[key] = record
        else:
            pending.append((key, row, identity))
    engines = [APIInferenceEngine(model=cfg['model'], base_url=endpoints[0], base_urls=endpoints,
        timeout=300, retries=2, max_tokens=budget, temperature=0, seed=0, extra_body=cfg['extra_body'])
        for budget in cfg['max_tokens_attempts']]
    update(phase='captioning', unique_media=len(unique), caption_complete=len(records),
           caption_cached=len(records), caption_failed=0)
    failed = []; concurrency = config.get('caption_concurrency', 32)
    # Bound decoded media and API payload memory to a small multiple of concurrency.
    with ThreadPoolExecutor(concurrency) as pool:
        for start in range(0, len(pending), concurrency*2):
            futures = {pool.submit(generate_caption, row, identity, key, cfg, engines): key
                       for key, row, identity in pending[start:start+concurrency*2]}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    record = future.result(); write_json(cache/f'{key}.json', record)
                    if record['status'] != 'ok':
                        raise ValueError('Caption did not finish normally after all attempts')
                    records[key] = record
                except Exception as exc:
                    failed.append({'key': key, 'error': f'{type(exc).__name__}: {exc}'})
                    write_json(output/job['name']/'caption_failures.json', failed)
                update(phase='captioning', unique_media=len(unique), caption_complete=len(records),
                       caption_failed=len(failed))
    if failed:
        raise RuntimeError(f'{len(failed)} captions failed; partial captions are not evaluated')
    bundle = {'schema_version': VERSION, 'config': cfg, 'media_manifest_digest': digest(manifest),
              'unique_media': len(unique), 'samples': [
                  {'sample_id': row['sample_id'], 'caption_key': sample_keys[row['sample_id']],
                   'caption': records[sample_keys[row['sample_id']]]['caption'],
                   'status': 'ok', 'finish_reason': records[sample_keys[row['sample_id']]]['finish_reason']}
                  for row in samples]}
    if destination.exists() and read_json(destination) != bundle:
        raise ValueError('Existing caption bundle changed; use a new output root')
    write_json(destination, bundle); CaptionStore(destination, manifest)
    update(phase='captions_complete', caption_complete=len(records), unique_media=len(unique),
           caption_file=str(destination), caption_digest=digest(bundle))
    return destination


def audit_result(result, manifest, captions_digest):
    expected = {r['sample_id'] for r in read_json(manifest)['samples']}; rows = result['results']
    if (len(rows) != len(expected) or {r['sample_id'] for r in rows} != expected
            or result['config']['observations']['captions']['digest'] != captions_digest):
        raise ValueError('QA coverage or shared caption identity mismatch')
    if any(r['status'] != 'ok' or r.get('finish_reason') not in {'stop', 'eos_token'}
           or not matches_constraint(r['raw_output'], r['response_constraint']) for r in rows):
        raise ValueError('Invalid or truncated final answers')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output-root', required=True)
    args = parser.parse_args(); config = read_json(args.config); out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'phase': 'starting', 'jobs': {}, 'tasks': {}, 'workers': {}}
    guard = threading.RLock(); child = None
    def save(**fields):
        with guard:
            state.update(fields); state['updated'] = time.time(); write_json(out/'status.json', state)
    def update(group, name, **fields):
        with guard:
            state[group].setdefault(str(name), {}).update(fields); save()
    pool = ReasonerPool(config, out, lambda gpu, **fields: update('workers', gpu, **fields))
    def interrupted(signum, frame):
        if child and child.poll() is None:
            try: os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError: pass
        pool.stop(); os._exit(128+signum)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    def summary():
        rows = []
        for key, result in state['tasks'].items():
            benchmark, variant, mode = key.split('/')
            rows.append({'benchmark': benchmark, 'variant': variant, 'mode': mode,
                         'phase': result['phase'], 'samples': result.get('samples'),
                         'score_100': result.get('metric', {}).get('score_100'),
                         'result': result.get('result'), 'error': result.get('error')})
        write_json(out/'summary.json', rows)
        if rows:
            with (out/'summary.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    with output_lock(out/'status.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != config:
            raise ValueError('Configuration changed; use a new output root')
        write_json(out/'config.json', config)
        tracked = ['core/captions.py', 'core/observations.py', 'core/inference.py', 'core/runner.py',
                   'scripts/run_caption_ablations.py', 'scripts/reasoner_pool.py']
        write_json(out/'process.json', {'pid': os.getpid(), 'start_time': process_identity(os.getpid()),
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'command': sys.argv,
            'toolkit_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in tracked}})
        try:
            for job in config['jobs']:
                rows = read_json(job['manifest'])['samples']
                if any(not r.get('media') for r in rows):
                    raise ValueError(f"Public RGB media is missing for {job['name']}")
                for variant in job['geometries']:
                    if not Path(variant['path']).is_file(): raise FileNotFoundError(variant['path'])
                update('jobs', job['name'], phase='queued', samples=len(rows))
                for variant, mode in [('shared_caption', 'caption')] + [(v['name'], 'caption_boxes') for v in job['geometries']]:
                    update('tasks', f"{job['name']}/{variant}/{mode}", phase='queued')
            save(phase='starting_servers'); endpoints = pool.start(); save(phase='running', endpoints=endpoints)
            for job in config['jobs']:
                name = job['name']
                try:
                    captions = prepare_captions(job, config, out, endpoints,
                        lambda **fields: update('jobs', name, **fields))
                    cap_digest = digest(read_json(captions))
                    for variant, mode in [({'name': 'shared_caption'}, 'caption')] + [(v, 'caption_boxes') for v in job['geometries']]:
                        key = f"{name}/{variant['name']}/{mode}"; target = out/key/'qa.json'
                        try:
                            update('tasks', key, phase='running', started=time.time())
                            command = [sys.executable, 'eval.py', '--benchmark', job['benchmark'], '--data', job['data'],
                                '--split', job['split'], '--model', config['llm_name'], '--base-urls', *endpoints,
                                '--temperature', '0', '--seed', '0', '--max-tokens', '4096', '--timeout', '300',
                                '--concurrency', str(config.get('qa_concurrency', 32)), '--batch-size', '64',
                                '--answer-format', 'native', '--observation-mode', mode, '--media-manifest', job['manifest'],
                                '--captions', str(captions), '--geometry-decimals', '4', '--max-prompt-chars', '600000',
                                '--max-image-pixels', str(config.get('max_image_pixels', 262144)), '--video-frames', '16',
                                '--extra-body', '{"chat_template_kwargs":{"enable_thinking":false}}',
                                '--run-label', key, '--output', str(target), '--resume', '--verbose']
                            if mode == 'caption_boxes': command += ['--geometry', variant['path']]
                            if job.get('dataset'): command += ['--dataset', job['dataset']]
                            target.parent.mkdir(parents=True, exist_ok=True)
                            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
                            with (target.parent/'eval.log').open('ab') as stream:
                                child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                            update('tasks', key, pid=child.pid)
                            if child.wait() != 0: raise RuntimeError(f'QA failed; inspect {target.parent}/eval.log')
                            result = read_json(target); audit_result(result, job['manifest'], cap_digest)
                            update('tasks', key, phase='complete', samples=result['num_samples'],
                                   metric=result['primary_metric'], result=str(target), completed=time.time())
                        except Exception as exc:
                            update('tasks', key, phase='failed', error=f'{type(exc).__name__}: {exc}')
                        summary()
                    phases = [v['phase'] for k, v in state['tasks'].items() if k.startswith(name+'/')]
                    update('jobs', name, phase='complete' if set(phases) == {'complete'} else 'failed')
                except Exception as exc:
                    update('jobs', name, phase='failed', error=f'{type(exc).__name__}: {exc}')
                    for key, value in state['tasks'].items():
                        if key.startswith(name+'/') and value['phase'] == 'queued':
                            update('tasks', key, phase='failed', error='Caption preparation failed')
                    summary()
            if not state['tasks'] or any(v['phase'] != 'complete' for v in state['tasks'].values()):
                raise RuntimeError('Some caption comparisons failed; valid cached work is resumable')
            save(phase='complete')
        except Exception as exc:
            save(phase='failed', error=f'{type(exc).__name__}: {exc}'); raise
        finally:
            if child and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM); child.wait(timeout=45)
            pool.stop()


if __name__ == '__main__':
    main()
