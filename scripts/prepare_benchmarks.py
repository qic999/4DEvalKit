"""Download pinned datasets and export complete, answer-free encoder inputs.

Preparation does not run an encoder or create benchmark scores. Each job has a
durable status and resumable sample records; missing media fails the job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time
import uuid
from urllib.parse import unquote, urlparse
import zipfile

from benchmark.loader import BenchmarkSession, media_manifest
from core.io import digest, read_json, source_signature, write_json
from core.runner import output_lock
from scripts.benchmark_registry import get_spec


def resolve_media_path(value, roots, repo_id=None):
    parsed = urlparse(str(value))
    if parsed.scheme in {'http', 'https'}:
        prefix = f'/datasets/{repo_id}/resolve/'
        if parsed.hostname != 'huggingface.co' or not repo_id or not parsed.path.startswith(prefix):
            raise ValueError(f'Unresolved remote media: {value}')
        relative = unquote(parsed.path[len(prefix):].split('/', 1)[1])
    else:
        relative = str(value)
    p = Path(relative)
    if p.is_absolute():
        return p.resolve(strict=True)
    if '..' in p.parts:
        raise ValueError(f'Unsafe media path: {value}')
    candidates = { (Path(root) / p).resolve() for root in roots if (Path(root) / p).is_file() }
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one local match for {value}; found {len(candidates)}')
    return candidates.pop()


def validate_media(value, roots, repo_id=None, cache=None):
    cache = {} if cache is None else cache
    if isinstance(value, list):
        if not value:
            raise ValueError('Empty media list')
        return [validate_media(x, roots, repo_id, cache) for x in value]
    if not isinstance(value, dict) or 'path' not in value:
        raise ValueError(f'Unexported media: {value}')
    path = resolve_media_path(value['path'], roots, repo_id)
    if path.stat().st_size == 0:
        raise ValueError(f'Empty media file: {path}')
    if str(path) not in cache:
        if path.suffix.lower() in {'.mp4', '.webm', '.avi', '.mov', '.mkv'}:
            import cv2
            cap = cv2.VideoCapture(str(path))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ok, _ = cap.read()
                if not ok or fps <= 0 or frames <= 0:
                    raise ValueError(f'Undecodable video or missing timing: {path}')
                info = {'fps': fps, 'frame_count': frames, 'duration_seconds': frames / fps,
                        'timing_source': 'video_container', 'validation': 'first_frame_decoded'}
            finally:
                cap.release()
        else:
            from PIL import Image
            with Image.open(path) as im:
                info = {'size': list(im.size)}
                im.verify()
        cache[str(path)] = info
    return {**value, 'path': str(path), **cache[str(path)]}


def extract_archive(archive, destination):
    destination = Path(destination).resolve()
    marker = destination / (Path(archive).name + '.extracted.json')
    identity = source_signature(archive)
    if marker.exists() and read_json(marker) == identity:
        return
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            target = (destination / entry.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f'Archive path escapes destination: {entry.filename}')
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == entry.file_size:
                continue
            import shutil
            temp = target.with_name(target.name + '.partial')
            with z.open(entry) as src, temp.open('wb') as dst:
                shutil.copyfileobj(src, dst)
            temp.replace(target)
    write_json(marker, identity)


def prepare_job(job, output_root):
    out = Path(output_root).resolve() / job['name']
    out.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'job': job['name'], 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    def status(phase, **extra):
        state.update(phase=phase, **extra)
        write_json(out / 'status.json', state)
    try:
        with output_lock(out / 'manifest.json'):
            status('acquiring_data')
            source = job.get('data')
            if 'repo_id' in job:
                from huggingface_hub import snapshot_download
                source = snapshot_download(job['repo_id'], repo_type='dataset', revision=job['revision'],
                    local_dir=str(out / 'source'), allow_patterns=job.get('allow_patterns'), max_workers=2)
            roots = [str(Path(source).parent if Path(source).is_file() else Path(source))]
            for archive in job.get('archives', []):
                extract_archive(archive, out / 'extracted')
                roots.extend([str(out / 'extracted'), str(out / 'extracted' / 'video')])
            identity = {'job': job, 'source': source_signature(source)}
            identity_file = out / 'source.json'
            if identity_file.exists() and read_json(identity_file) != identity:
                raise ValueError('Preparation source changed; choose a fresh output root')
            write_json(identity_file, identity)
            for split in job.get('splits', [job.get('split')]):
                spec = get_spec(job['benchmark'])
                split = split or spec.split
                dest = out / split
                session = BenchmarkSession(spec, data=source, split=split, batch_size=8)
                status('exporting_media', split=split, completed_samples=0, total_samples=session.selected_count)
                records, cache = [], {}
                for batch in session.batches():
                    for example in batch:
                        saved = dest / 'records' / (example['sample_id'].replace(':', '_') + '.json')
                        fingerprint = digest({'source': identity, 'sample': example['answer_fingerprint'],
                                              'question': example['question'], 'id': example['sample_id']})
                        if saved.exists():
                            record = read_json(saved)
                            if record['fingerprint'] != fingerprint:
                                raise ValueError(f"Sample changed: {example['sample_id']}")
                            row = record['input']
                        else:
                            row = media_manifest(example, dest / 'media' / uuid.uuid4().hex)
                        if not row['media']:
                            raise ValueError(f"No encoder media: {row['sample_id']}")
                        row['media'] = {k: validate_media(v, roots, job.get('repo_id'), cache)
                                        for k,v in row['media'].items()}
                        write_json(saved, {'fingerprint': fingerprint, 'input': row})
                        records.append(row)
                    status('exporting_media', completed_samples=len(records))
                write_json(dest / 'manifest.json', {'benchmark': spec.name, 'split': split,
                    'source': identity, 'samples': records, 'num_samples': len(records),
                    'stage': 'encoder_inputs_ready', 'geometry_generated': False})
            status('encoder_inputs_ready', geometry_generated=False,
                   next_step='Generate predicted proposals, 3D geometry and (for videos) aligned tracks before LLM evaluation')
        return state
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    jobs = read_json(args.config)['jobs']
    if len({j['name'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate preparation job names')
    states = {}
    with output_lock(Path(args.output_root) / 'status.json'):
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = {pool.submit(prepare_job, j, args.output_root): j['name'] for j in jobs}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    states[name] = future.result()
                except Exception as exc:
                    states[name] = {'phase': 'failed', 'error': str(exc)}
                write_json(Path(args.output_root) / 'status.json', {'pid': os.getpid(), 'jobs': states})
                print(name, states[name], flush=True)
    if any(s['phase'] == 'failed' for s in states.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
