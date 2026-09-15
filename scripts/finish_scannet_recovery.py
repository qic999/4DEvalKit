"""Wait for a recorded final encoder worker, then finish metrics and QA.

Uses process start times to avoid confusing reused PIDs with this run's workers.
Model truncations remain scored failures and are explicitly reported by audit.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from core.io import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--encoder-process', required=True)
    p.add_argument('--wait-controller-pid', type=int, required=True)
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    worker = read_json(args.encoder_process)
    state = {'pid': os.getpid(), 'encoder_pid': worker['pid'], 'phase': 'waiting_encoder'}
    def status(phase, **extra):
        state.update(phase=phase, **extra)
        write_json(run / 'recovery/final_status.json', state)
    try:
        targets = [(worker['pid'], process_identity(worker['pid'])),
                   (args.wait_controller_pid, process_identity(args.wait_controller_pid))]
        deadline = time.monotonic() + 6 * 3600
        for pid, identity in targets:
            while identity is not None and process_identity(pid) == identity:
                if time.monotonic() > deadline:
                    raise TimeoutError('Recovery prerequisite exceeded six hours')
                status('waiting_encoder' if pid == worker['pid'] else 'waiting_small_qa')
                time.sleep(15)
        scenes = set(read_json(run / 'manifest.json')['scene_ids'])
        for v in read_json(args.config)['variants']:
            pred = run / 'BoxDet/pred' / (v['name'] + '_scannet_val88_gt2d_all')
            if {x.parent.name for x in pred.glob('*/done.flag')} != scenes:
                raise RuntimeError(f"{v['name']}: encoder exited with incomplete scenes; see {worker['log']}")
        status('postprocessing')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.run_scannet_models',
            '--config', args.config, '--output-root', str(run), '--resume', '--postprocess-only'], cwd=ROOT, check=True)
        status('auditing')
        subprocess.run([sys.executable, '-u', '-m', 'scripts.watch_scannet_results',
            '--run-dir', str(run), '--allow-invalid-completions'], cwd=ROOT, check=True)
        status('complete', summary=str(run / 'summary.csv'))
    except Exception as exc:
        status('failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
