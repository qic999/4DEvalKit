"""Run real temporal QA smoke/full suites with predicted geometry and RGB arms.

GPU workers generate proposals and encode per-camera tracks. Identical reasoner
replicas then evaluate boxes, RGB, and RGB+boxes with constrained final answers.
Durable status.json distinguishes completed scores from failed/incomplete jobs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from core.io import read_json, write_json
from core.runner import output_lock
from scripts.gpu_lease import acquire_gpu
from scripts.reasoner_pool import ReasonerPool, wait_for_free_gpu
from scripts.run_checkpoint_evaluation import audit_result
from scripts.run_checkpoint_evaluation import merge_geometry

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); c = read_json(args.config)
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    state = {'phase': 'encoding', 'pid': os.getpid(), 'jobs': {}, 'workers': {}}
    guard = threading.RLock(); children = []; pool = None
    stopping = threading.Event()

    def update(group, name, **fields):
        with guard:
            state[group].setdefault(str(name), {}).update(fields)
            state['updated'] = time.time(); write_json(out/'status.json', state)

    def run(command, log, gpu=None):
        if stopping.is_set(): raise RuntimeError('Suite is stopping')
        env = dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                   PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONPATH=str(ROOT))
        if gpu is not None: env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open('ab') as stream:
            with guard:
                if stopping.is_set(): raise RuntimeError('Suite is stopping')
                child = subprocess.Popen(list(map(str, command)), cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                children.append(child)
        if child.wait(): raise RuntimeError(f'Process failed; inspect {log}')

    def stop(signum=None, frame=None):
        stopping.set()
        for child in children:
            if child.poll() is None:
                try: os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        if pool: pool.stop()
        if signum: raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    jobs = queue.Queue()
    shards = int(c.get('encoding_shards', 1))
    if shards < 1: raise ValueError('encoding_shards must be positive')
    for job in c['jobs']:
        for name in (['annotations.json','manifest.json'] if c.get('modes', ['boxes']) else ['manifest.json']):
            if not (Path(job['prepared'])/name).is_file(): raise FileNotFoundError(Path(job['prepared'])/name)
        original = read_json(Path(job['prepared'])/'manifest.json')
        if shards == 1:
            jobs.put(dict(job, shard=None, manifest=str(Path(job['prepared'])/'manifest.json')))
        else:
            # Keep adjacent questions for a video together so frame inference
            # can be reused. Never split video sequences across encoder calls.
            groups = {}
            from core.io import digest
            for row in original['samples']:
                key = digest({'media': row['media'], 'metadata': row.get('input_metadata', {})})
                groups.setdefault(key, []).append(row)
            partitions = [[] for _ in range(shards)]
            for i, rows in enumerate(groups.values()): partitions[i % shards].extend(rows)
            for i, rows in enumerate(partitions):
                if not rows: continue
                path = out/job['benchmark']/'shards'/str(i)/'manifest.json'
                write_json(path, dict(original, samples=rows, num_samples=len(rows)))
                jobs.put(dict(job, shard=i, manifest=str(path)))

    def encode_worker(gpu):
        lease = acquire_gpu(gpu)
        try:
            wait_for_free_gpu(gpu, lambda gpu, **kw: update('workers', gpu, **kw))
            while True:
                if stopping.is_set(): return
                try: job = jobs.get_nowait()
                except queue.Empty: return
                name = job['benchmark']; folder = out/name
                if job['shard'] is not None:
                    name += f"/shard_{job['shard']}"; folder = folder/'shards'/str(job['shard'])
                manifest = Path(job['manifest'])
                try:
                    update('jobs', name, phase='proposals', gpu=gpu)
                    if not (folder/'proposals/index.json').is_file() or not read_json(folder/'proposals/index.json')['complete']:
                        run([c['llm_python'], '-m', 'scripts.generate_media_proposals', '--manifest', manifest,
                             '--output', folder/'proposals', '--model', c['detector_model'],
                             '--video-frames', job.get('video_frames',16), '--max-objects', c.get('max_objects',40)],
                             folder/'proposals.log', gpu)
                    update('jobs', name, phase='encoding', gpu=gpu)
                    run([c['encoder_python'], '-m', 'scripts.encode_media_geometry', '--manifest', manifest,
                         '--proposals', folder/'proposals', '--model-repo', c['model_repo'], '--checkpoint', c['checkpoint'],
                         '--profile', c['profile'], '--resolution', c['resolution'], '--spatial-resolution', c['spatial_resolution'],
                         '--model-image-size', c['model_image_size'], '--output', folder/'geometry'], folder/'encoder.log', gpu)
                    update('jobs', name, phase='geometry_ready', gpu=gpu)
                except Exception as exc: update('jobs', name, phase='failed', error=str(exc))
                finally: jobs.task_done()
        finally: lease.close()

    with output_lock(out/'suite.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != c:
            raise ValueError('Suite config changed; choose a new output directory')
        write_json(out/'config.json', c)
        try:
            with ThreadPoolExecutor(len(c['gpus'])) as workers: list(workers.map(encode_worker, c['gpus']))
            if shards > 1:
                for job in c['jobs']:
                    name = job['benchmark']
                    try:
                        paths = sorted((out/name/'shards').glob('*/geometry/geometry.json'))
                        expected = [r['sample_id'] for r in read_json(Path(job['prepared'])/'manifest.json')['samples']]
                        merged = merge_geometry(paths, expected)
                        write_json(out/name/'geometry/geometry.json', merged)
                        update('jobs', name, phase='geometry_ready')
                    except Exception as exc: update('jobs', name, phase='failed', error=str(exc))
            endpoints = []
            if c.get('modes', ['boxes','rgb','rgb_boxes']):
                state['phase'] = 'starting_reasoners'
                pc = dict(c, gpus=c.get('reasoner_gpus', c['gpus']))
                pool = ReasonerPool(pc, out, lambda gpu, **kw: update('workers',gpu,**kw))
                endpoints = pool.start()
            state['phase'] = 'reasoning'
            tasks = []
            for job in c['jobs']:
                if state['jobs'][job['benchmark']]['phase'] == 'geometry_ready':
                    tasks.extend((job, mode) for mode in c.get('modes', ['boxes','rgb','rgb_boxes']))

            def qa(task):
                job, mode = task; name=job['benchmark']; key=f'{name}/{mode}'
                prepared=Path(job['prepared']); target=out/name/'qa'/f'{mode}.json'
                try:
                    update('jobs',key,phase='reasoning')
                    command=[sys.executable, 'eval.py', '--benchmark',name,'--data',prepared/'annotations.json',
                        '--model',c['llm_name'],'--base-urls',*endpoints,'--answer-format','native',
                        '--max-tokens','4096','--timeout','300','--concurrency',str(c.get('qa_concurrency',8)),
                        '--batch-size',str(c.get('qa_batch_size',32)),'--observation-mode',mode,'--media-manifest',prepared/'manifest.json',
                        '--video-frames',str(job.get('video_frames',16)),'--max-image-pixels',str(c.get('max_image_pixels',262144)),
                        '--max-prompt-chars','600000','--geometry-decimals','4','--temperature','0','--seed','0',
                        '--extra-body','{"chat_template_kwargs":{"enable_thinking":false}}',
                        '--output',target,'--resume']
                    if mode != 'rgb': command += ['--geometry',out/name/'geometry/geometry.json']
                    run(command, target.with_suffix('.log'))
                    result=read_json(target); expected=[r['sample_id'] for r in read_json(prepared/'manifest.json')['samples']]
                    audit_result(result,expected,mode)
                    update('jobs',key,phase='complete',samples=result['num_samples'],
                        score_100=result['primary_metric']['score_100'],result=str(target))
                except Exception as exc: update('jobs',key,phase='failed',error=str(exc))
            with ThreadPoolExecutor(c.get('parallel_qa_jobs',2)) as workers: list(workers.map(qa,tasks))
            state['phase']='failed' if any(v['phase']=='failed' for v in state['jobs'].values()) else 'complete'
            write_json(out/'status.json',state)
            write_json(out/'suite.json',state)
        finally: stop()


if __name__ == '__main__':
    main()
