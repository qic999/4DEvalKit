"""Supervise adopted ScanNet workers and resume OOM shards without re-encoding.

The explicit plan records existing PIDs/start times and original commands. Only
failed OOM workers receive one retry, with completed tracker masks on CPU.
Healthy workers are never signalled. After the original controller exits, run
merge/metrics/QA and the result audit using the same immutable configuration.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from scripts.run_scannet_models import file_info, write_json

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    """Linux start tick prevents mistaking a reused PID for an adopted worker."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(record):
    return bool(record.get('pid') and record.get('start_tick')
                and process_identity(record['pid']) == record['start_tick'])


def failed_clip(log):
    """Only retry a logged CUDA OOM; never skip a failed frame or other error."""
    segment = log.rsplit('[4deval-recovery]', 1)[-1]
    if 'OutOfMemoryError:' not in segment:
        raise RuntimeError('Worker stopped without completed scenes or a retryable CUDA OOM')
    infer = re.findall(r'\[infer\] scene=(\S+) clip=(\d+)/(\d+)', segment)
    if not infer:
        raise RuntimeError('OOM occurred before an identifiable clip')
    scene, index, total = infer[-1]
    return scene, int(index) - 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    args = p.parse_args()
    plan = json.loads(Path(args.plan).read_text())
    run = Path(plan['run_dir']).resolve()
    recovery = run / 'recovery'
    recovery.mkdir(exist_ok=True)
    lock = (recovery / 'supervisor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((run / 'manifest.json').read_text())
    config = manifest['config']
    config_path = Path(plan['config']).resolve()
    actual_config = json.loads(config_path.read_text())
    fingerprint = hashlib.sha256(json.dumps(actual_config, sort_keys=True).encode()).hexdigest()
    if fingerprint != manifest['config_sha256']:
        raise ValueError('Recovery config differs from the original run')
    for name, checkpoint in manifest['checkpoints'].items():
        if file_info(checkpoint['path']) != checkpoint:
            raise ValueError(f'{name}: checkpoint changed')
    parity = json.loads(Path(plan['parity_report']).read_text())
    if not parity.get('passed'):
        raise ValueError('Tracker mask offload parity check must pass before recovery')
    env = dict(os.environ, PYTHONUNBUFFERED='1',
               BOX_DATA_PATH=config['data_root'], BOX_OUTPUT_PATH=str(run),
               FOURDEVAL_ENCODER_PYTHON=config['encoder_base_python'],
               FOURDEVAL_BOX_ONLY_FUSION='1' if config.get('box_only_detector_fusion') else '0',
               FOURDEVAL_TRACKER_MASK_OFFLOAD='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
    jobs = plan['workers']
    processes = {}
    state = dict(pid=os.getpid(), start_tick=process_identity(os.getpid()),
                 phase='supervising_encoders', workers=jobs,
                 optimization='Completed tracker masks on CPU; temporal memory and 3D fields unchanged',
                 encoder_entry_sha256=hashlib.sha256((ROOT / 'scripts/encoder_entry.py').read_bytes()).hexdigest())

    def save():
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(recovery / 'status.json', state)

    try:
        while True:
            for job in jobs:
                key = f'{job["variant"]}:{job["gpu"]}'
                if key in processes:
                    processes[key].poll()  # reap an owned worker before checking /proc
                if alive(job):
                    job['phase'] = 'recovery_running' if job.get('attempts') else 'original_running'
                    continue
                scenes = Path(job['shard']).read_text().split()
                pred = run / 'BoxDet/pred' / (job['variant'] + '_scannet_val88_gt2d_all')
                if all((pred / scene / 'done.flag').is_file() for scene in scenes):
                    job['phase'] = 'complete'
                    continue
                if job.get('attempts', 0) >= 1:
                    raise RuntimeError(f'{key}: retry exited before shard completion; see {job["log"]}')
                log = Path(job['log'])
                scene, clip = failed_clip(log.read_text(errors='replace'))
                usage = subprocess.check_output(['nvidia-smi', '-i', str(job['gpu']),
                    '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                if int(usage.strip()) > int(config.get('max_existing_gpu_memory_mb', 1024)):
                    job['phase'] = 'waiting_for_gpu'
                    continue
                command = [*job['command'], '--skip-done', '--start-clip', str(clip)]
                with log.open('a') as stream:
                    stream.write(f'\n[4deval-recovery] attempt=1 scene={scene} start_clip={clip} mask_offload=1\n')
                    stream.flush()
                    proc = subprocess.Popen(command, cwd=config['model_repo'],
                        env=dict(env, CUDA_VISIBLE_DEVICES=str(job['gpu'])),
                        stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                        start_new_session=True)
                processes[key] = proc
                job.update(pid=proc.pid, start_tick=process_identity(proc.pid), attempts=1,
                           phase='recovery_running', resume_clip=clip)
                print(f'Restarted {key} at clip {clip}, PID {proc.pid}', flush=True)
            save()
            if all(job['phase'] == 'complete' for job in jobs):
                break
            time.sleep(30)
        state['phase'] = 'waiting_for_original_controller'
        save()
        while alive(plan['controller']):
            time.sleep(30)
        # The previous observer exits when its controller fails. Wait for it so
        # it cannot race the new observer's summary writes.
        while alive(plan.get('observer', {})):
            time.sleep(30)
        status = json.loads((run / 'status.json').read_text())
        if not all(v['phase'] == 'complete' for v in status['variants'].values()):
            state['phase'] = 'postprocessing_and_reasoning'
            save()
            with (recovery / 'postprocess.log').open('a') as stream:
                proc = subprocess.Popen([config['eval_python'], '-u', '-m', 'scripts.run_scannet_models',
                    '--config', str(config_path), '--output-root', str(run), '--resume', '--postprocess-only'],
                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, start_new_session=True)
                state['postprocess_pid'] = proc.pid
                save()
                if proc.wait():
                    raise RuntimeError('Postprocessing failed; see recovery/postprocess.log')
        state['phase'] = 'auditing_results'
        save()
        with (recovery / 'summary.log').open('a') as stream:
            subprocess.run([config['eval_python'], '-u', '-m', 'scripts.watch_scannet_results',
                            '--run-dir', str(run)], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        state['phase'] = 'complete'
        save()
    except Exception as exc:
        state.update(phase='failed', error=str(exc))
        save()
        # Leave healthy original and recovered workers running; retain all data.
        raise


if __name__ == '__main__':
    main()
