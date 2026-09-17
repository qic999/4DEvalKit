"""After a caption suite exits, rerun incomplete splits with a recorded caption penalty.

All captions and all three arms of an affected split are regenerated in a new
output root. Original results are preserved; combined summaries name their source.
"""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
import time

from core.io import read_json, write_json
from core.runner import output_lock
from scripts.finish_scannet_recovery import process_identity

ROOT = Path(__file__).resolve().parents[1]


def incomplete_jobs(config, status):
    pending = []
    for job in config['jobs']:
        keys = [f"{job['name']}/shared_caption/caption"] + [
            f"{job['name']}/{v['name']}/caption_boxes" for v in job['geometries']]
        if any(status.get('tasks', {}).get(k, {}).get('phase') != 'complete' for k in keys):
            pending.append(job)
    return pending


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--caption-repetition-penalty', type=float, default=1.1)
    args = parser.parse_args()
    source = Path(args.source_root).resolve(); out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(source/'config.json'); prerequisite = read_json(source/'process.json')
    if not prerequisite.get('start_time'):
        raise ValueError('Source controller must have a recorded process identity')
    state = {'pid': os.getpid(), 'start_time': process_identity(os.getpid()),
             'source_process': prerequisite, 'source_root': str(source)}
    def status(phase, **fields):
        state.update(phase=phase, updated=time.time(), **fields); write_json(out/'recovery_status.json', state)
    with output_lock(out/'recovery_status.json'):
        try:
            status('waiting_for_source')
            deadline = time.monotonic()+72*3600
            while process_identity(prerequisite['pid']) == prerequisite['start_time']:
                if time.monotonic() > deadline:
                    raise TimeoutError('Source controller did not exit within 72 hours')
                time.sleep(20)
            jobs = incomplete_jobs(config, read_json(source/'status.json'))
            if jobs:
                config = {**config, 'jobs': jobs, 'caption_repetition_penalty': args.caption_repetition_penalty}
                config_path = out/'recovery_config.json'; write_json(config_path, config)
                status('recovering', jobs=[j['name'] for j in jobs],
                       caption_repetition_penalty=args.caption_repetition_penalty)
                command = [sys.executable, '-u', '-m', 'scripts.run_caption_ablations',
                           '--config', str(config_path), '--output-root', str(out/'rerun')]
                with (out/'rerun.log').open('ab') as stream:
                    child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                status('recovering', child_pid=child.pid, child_start_time=process_identity(child.pid))
                if child.wait() != 0:
                    raise RuntimeError('Recovery remains incomplete; inspect rerun/status.json')
            replaced = {j['name'] for j in jobs}
            rows = [{**r, 'run_source': 'original'} for r in read_json(source/'summary.json')
                    if r['benchmark'] not in replaced]
            if jobs:
                rows += [{**r, 'run_source': 'caption_repetition_penalty_recovery'}
                         for r in read_json(out/'rerun/summary.json')]
            expected = sum(1+len(j['geometries']) for j in read_json(source/'config.json')['jobs'])
            keys = {(r['benchmark'], r['variant'], r['mode']) for r in rows}
            if len(rows) != expected or len(keys) != expected or any(r['phase'] != 'complete' for r in rows):
                raise ValueError('Combined recovery coverage is incomplete')
            write_json(out/'combined_summary.json', rows)
            with (out/'combined_summary.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            status('complete', tasks=len(rows), responses=sum(r['samples'] for r in rows))
        except Exception as exc:
            status('failed', error=f'{type(exc).__name__}: {exc}'); raise


if __name__ == '__main__':
    main()
