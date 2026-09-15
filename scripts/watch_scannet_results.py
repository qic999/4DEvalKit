"""Audit fresh val88 results, replay the project metric and write a comparison CSV.

This observer uses no GPUs and never reruns the encoder or reasoning LLM.
It exits if the owning pipeline fails instead of waiting forever for results.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from core.io import write_json
from scripts.summarize_results import rows_from_paths

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--allow-invalid-completions', action='store_true',
                        help='Keep truncated/empty model responses as failures; never accept missing/API errors')
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    config = manifest['config']
    expected = manifest['question_count']
    names = [v['name'] for v in config['variants']]
    done, paths = set(), []
    while len(done) != len(names):
        state = json.loads((run / 'status.json').read_text())
        for name in names:
            if name in done:
                continue
            source = run / 'qa' / (name + '.json')
            phase = state['variants'][name]['phase']
            if phase == 'failed':
                raise RuntimeError(f'{name} failed: {state["variants"][name].get("error")}')
            if not source.is_file() or phase != 'complete':
                continue
            result = json.loads(source.read_text())
            if result['num_samples'] != expected or len({x['sample_id'] for x in result['results']}) != expected:
                raise RuntimeError(f'{name}: incomplete or duplicate QA coverage')
            counts = result['status_counts']
            allowed = {'ok', 'invalid_completion'} if args.allow_invalid_completions else {'ok'}
            if sum(counts.values()) != expected or set(counts) - allowed:
                raise RuntimeError(f'{name}: invalid/request failure counts {counts}; resume needed')
            target = run / 'qa_project' / (name + '.json')
            log = run / 'logs' / (name + '_project_replay.log')
            with log.open('a') as stream:
                subprocess.run([config['eval_python'], 'eval.py', '--benchmark', 'VSI-Bench',
                    '--data', config['qa_json'], '--dataset', 'scannet', '--predictions', str(source),
                    '--vsi-metric-protocol', 'project', '--model', config['llm_name'],
                    '--run-label', name + '_gt2d_pred3d_gtpose', '--output', str(target), '--resume'],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
            paths.extend([source, target])
            rows = list(rows_from_paths(paths))
            write_json(run / 'summary.json', {'completed_variants': sorted(done | {name}), 'rows': rows})
            subprocess.run([config['eval_python'], '-m', 'scripts.summarize_results',
                            *map(str, paths), '--output', str(run / 'summary.csv')], cwd=ROOT, check=True)
            done.add(name)
            print(f'Audited {name}: {expected} covered questions; status counts {counts}; both metric protocols saved', flush=True)
        if len(done) != len(names):
            try:
                os.kill(int(state['pid']), 0)
            except ProcessLookupError as exc:
                # The controller may have completed between the status read
                # above and this liveness check. Re-read before declaring loss.
                fresh = json.loads((run / 'status.json').read_text())
                if all(v['phase'] == 'complete' for v in fresh['variants'].values()):
                    continue
                raise RuntimeError('Model pipeline exited before required results were complete') from exc
            time.sleep(30)
    print(f'Finished: {run / "summary.csv"}', flush=True)


if __name__ == '__main__':
    main()
