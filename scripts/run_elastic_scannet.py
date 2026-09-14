"""Use available GPUs for a queue of remaining Full scenes, then score both models.

Existing Small workers finish normally. Their GPUs join the Full queue as they
become free. Jobs preserve the native shard indices and per-scene output paths.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from scripts.elastic_queue import locked_queue
from scripts.recover_scannet_workers import alive, process_identity
from scripts.run_scannet_models import write_json, file_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', required=True)
    args = p.parse_args()
    queue_path = Path(args.queue).resolve()
    initial = json.loads(queue_path.read_text())
    run = Path(initial['run_dir'])
    directory = run / 'elastic'
    lock = (directory / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = initial['config']
    manifest = json.loads((run / 'manifest.json').read_text())
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != manifest['config_sha256']:
        raise ValueError('Evaluation configuration changed')
    for info in manifest['checkpoints'].values():
        if file_info(info['path']) != info:
            raise ValueError('Checkpoint changed')
    if not json.loads((directory / 'interpolation_probe/parity.json').read_text()).get('passed'):
        raise ValueError('Dense-clip recovery validation has not passed')
    status_path = directory / 'status.json'
    old = json.loads(status_path.read_text()) if status_path.exists() else {}
    state = dict(pid=os.getpid(), start_tick=process_identity(os.getpid()), phase='encoding',
                 workers=old.get('workers', []), adopted_workers=initial['adopted_workers'],
                 gpu_pool=initial['gpu_pool'], queue=str(queue_path))
    children = {}
    env = dict(os.environ, BOX_DATA_PATH=config['data_root'], BOX_OUTPUT_PATH=str(run),
               FOURDEVAL_ENCODER_PYTHON=config['encoder_base_python'],
               PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')

    def save():
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        state['completed_scenes'] = {
            v['name']: sum(1 for _ in (run / 'BoxDet/pred' /
                (v['name'] + '_scannet_val88_gt2d_all')).glob('*/done.flag'))
            for v in config['variants']}
        write_json(status_path, state)

    try:
        while True:
            for proc in children.values():
                proc.poll()
            with locked_queue(queue_path) as queue:
                for job in queue['jobs']:
                    if job['phase'] == 'running' and not alive(job):
                        job.update(phase='failed', error='Worker exited during this scene')
                    if job['phase'] != 'failed' or job.get('attempts', 0) >= 2:
                        continue
                    log = Path(job['log']).read_text(errors='replace')
                    segment = log.rsplit('[elastic-job]', 1)[-1]
                    if not any(x in segment for x in ['OutOfMemoryError:', 'less than INT_MAX']):
                        continue
                    writes = re.findall(r'\[write\] scene=\S+ clip=(\d+)', segment)
                    if writes:
                        job['start'] = max(job['start'], int(writes[-1]) + 1)
                    if job['start'] < job['stop']:
                        job.update(phase='pending', retry_reason=job.get('error'))
                counts = Counter(j['phase'] for j in queue['jobs'])
            state['job_counts'] = dict(counts)
            active = [x for x in state['workers'] if alive(x)]
            adopted = [x for x in state['adopted_workers'] if alive(x)]
            busy = {x['gpu'] for x in active + adopted}
            usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            memory = {int(a.strip()): int(b.strip()) for a, b in
                      (line.split(',') for line in usage.splitlines())}
            for gpu in state['gpu_pool']:
                if not counts['pending'] or gpu in busy or memory[gpu] > 1024:
                    continue
                # Workers claim under a file lock after model initialization.
                serial = 1 + sum(w['gpu'] == gpu for w in state['workers'])
                log = directory / f'gpu_{gpu}_worker_{serial:02d}.log'
                command = [config['encoder_base_python'], '-u', '-m', 'scripts.elastic_encoder',
                           '--queue', str(queue_path), '--gpu', str(gpu), '--log', str(log)]
                with log.open('a') as stream:
                    proc = subprocess.Popen(command, cwd=ROOT,
                        env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)), stdout=stream,
                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                children[proc.pid] = proc
                state['workers'].append(dict(pid=proc.pid, start_tick=process_identity(proc.pid),
                    gpu=gpu, variant=initial['variant'], log=str(log)))
                print(f'GPU {gpu}: Full queue worker PID {proc.pid}', flush=True)
                save()
            save()
            if not counts['pending'] and not counts['running'] and not active:
                if counts['failed']:
                    raise RuntimeError(f'{counts["failed"]} scene jobs failed; see queue and worker logs')
                if not adopted and all(n == 88 for n in state['completed_scenes'].values()):
                    break
                if not adopted:
                    raise RuntimeError('Original Small workers ended with incomplete scene coverage')
            time.sleep(15)
        state['phase'] = 'waiting_for_native_small_metrics'
        save()
        while alive(initial['native_small_controller']):
            time.sleep(15)
        state['phase'] = 'postprocessing_and_reasoning'
        save()
        with (directory / 'postprocess.log').open('a') as stream:
            proc = subprocess.Popen([config['eval_python'], '-u', '-m', 'scripts.run_scannet_models',
                '--config', initial['config_path'], '--output-root', str(run), '--resume',
                '--postprocess-only'], cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            state['postprocess_pid'] = proc.pid
            save()
            if proc.wait():
                raise RuntimeError('Postprocessing failed; see elastic/postprocess.log')
        state['phase'] = 'auditing_results'
        save()
        with (directory / 'summary.log').open('a') as stream:
            subprocess.run([config['eval_python'], '-u', '-m', 'scripts.watch_scannet_results',
                            '--run-dir', str(run)], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        state['phase'] = 'complete'
        save()
    except Exception as exc:
        state.update(phase='failed', error=str(exc))
        save()
        raise


if __name__ == '__main__':
    main()
