"""Read model/QA progress from a run directory without loading model weights."""
import argparse
import json
import os
from pathlib import Path
import re

from scripts.recover_scannet_workers import alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir')
    args = parser.parse_args()
    root = Path(args.run_dir)
    state = json.loads((root / 'status.json').read_text())
    final_path = root / 'recovery/final_status.json'
    if final_path.is_file():
        from scripts.finish_scannet_recovery import process_identity
        final = json.loads(final_path.read_text())
        final['alive'] = process_identity(final['pid']) is not None
        worker_path = root / 'recovery/full_final39_process.json'
        ranges_path = root / 'recovery/range_controller_process.json'
        if ranges_path.exists() and json.loads(ranges_path.read_text())['pid'] == final.get('encoder_pid'):
            worker_path = ranges_path
        if worker_path.is_file():
            worker = json.loads(worker_path.read_text())
            text = Path(worker['log']).read_text(errors='replace')
            worker.update(alive=process_identity(worker['pid']) is not None,
                          completed_clips=len(re.findall(r'\[write\]', text)))
            if worker_path == ranges_path:
                ranges_status = Path(worker['log']).parent/'status.json'
                if ranges_status.exists():
                    worker['ranges'] = json.loads(ranges_status.read_text())
                    worker['completed_clips'] = sum(len(re.findall(r'\[write\]',Path(w['log']).read_text()))
                                                    for w in worker['ranges']['workers'])
            final['encoder_worker'] = worker
        state['final_recovery'] = final
    elastic_path = root / 'elastic/status.json'
    elastic = json.loads(elastic_path.read_text()) if elastic_path.exists() else None
    if elastic:
        state['elastic'] = dict(elastic, alive=alive(elastic))
        queue = json.loads(Path(elastic['queue']).read_text())
        state['elastic']['active_jobs'] = [{k: j.get(k) for k in
            ['id', 'scene', 'start', 'stop', 'pid', 'gpu', 'phase', 'log', 'error', 'attempts']}
            for j in queue['jobs'] if j['phase'] in ('running', 'failed')]
    recovery_path = root / 'recovery/status.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.is_file() else None
    if recovery:
        state['recovery'] = {k: v for k, v in recovery.items() if k != 'workers'}
        state['recovery']['alive'] = alive(recovery)
    try:
        os.kill(int(state['pid']), 0)
        state['controller_alive'] = True
    except ProcessLookupError:
        state['controller_alive'] = False
    for name, variant in state['variants'].items():
        worker_root = root / 'runs' / (name + '_scannet_val88_gt2d_all') / 'logs'
        pred_root = root / 'BoxDet/pred' / (name + '_scannet_val88_gt2d_all')
        variant['completed_scenes'] = sum(1 for _ in pred_root.glob('*/done.flag'))
        workers = []
        for path in sorted(worker_root.glob('gpu_*.log')):
            text = path.read_text(errors='replace')
            infer = re.findall(r'\[infer\] scene=(\S+) clip=(\d+)/(\d+)', text)
            writes = re.findall(r'\[write\].*?predictions=(\d+) pose_frames=(\d+)', text)
            ckpt = re.findall(r'\[ckpt\] loaded[^\n]+', text)
            current_text = text.rsplit('[4deval-recovery]', 1)[-1]
            errors = [line for line in current_text.splitlines() if
                      'OutOfMemoryError:' in line or line.startswith(('RuntimeError:', 'FileNotFoundError:'))]
            worker = {'gpu': int(path.stem.split('_')[1]), 'log': str(path),
                      'checkpoint_load': ckpt[-1] if ckpt else 'not yet confirmed',
                      'clips_written_in_current_log': len(writes),
                      'object_observations_written': sum(int(x[0]) for x in writes),
                      'pose_frames_written': sum(int(x[1]) for x in writes)}
            if errors:
                worker['error'] = errors[-1]
            if recovery:
                adopted = next((x for x in recovery['workers']
                                if x['variant'] == name and x['gpu'] == worker['gpu']), None)
                if adopted:
                    worker.update(pid=adopted.get('pid'), alive=alive(adopted),
                                  phase=adopted.get('phase'), recovery_attempts=adopted.get('attempts', 0))
                    if adopted.get('resume_clip') is not None:
                        worker['resume_clip_zero_based'] = adopted['resume_clip']
            if infer:
                scene, index, total = infer[-1]
                worker.update(scene=scene, dataset_clip_index=int(index), total_dataset_clips=int(total))
            workers.append(worker)
        if elastic:
            if elastic['phase'] == 'encoding' and alive(elastic):
                variant['phase'] = 'encoder_elastic' if name == queue['variant'] else 'encoder_original'
            for original in workers:
                if name == queue['variant']:
                    original['phase'] = 'superseded'
                    original['alive'] = False
            for record in elastic['workers']:
                if record['variant'] != name:
                    continue
                path = Path(record['log'])
                log_text = path.read_text(errors='replace') if path.exists() else ''
                writes = re.findall(r'\[write\].*?predictions=(\d+) pose_frames=(\d+)', log_text)
                infer = re.findall(r'\[infer\] scene=(\S+) clip=(\d+)/(\d+)', log_text)
                worker = dict(record, alive=alive(record), phase='elastic',
                              clips_written_in_current_log=len(writes),
                              object_observations_written=sum(int(x[0]) for x in writes))
                if infer:
                    worker.update(scene=infer[-1][0], dataset_clip_index=int(infer[-1][1]),
                                  total_dataset_clips=int(infer[-1][2]))
                workers.append(worker)
        variant['workers'] = workers
        journal = root / 'qa' / (name + '.jsonl')
        variant['qa_saved'] = sum(1 for _ in journal.open()) if journal.exists() else 0
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
