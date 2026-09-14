"""Keep a Full model loaded while claiming remaining scenes from a shared queue."""
import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

from scripts.elastic_queue import claim_job, finish_job
from scripts.encoder_entry import (install_box_only_detector_fusion,
                                  install_tracker_mask_offload, install_large_interpolation)
from scripts.recover_scannet_workers import process_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--gpu', required=True, type=int)
    parser.add_argument('--log', required=True)
    options = parser.parse_args()
    queue = json.loads(Path(options.queue).read_text())
    config = queue['config']
    variant = next(v for v in config['variants'] if v['name'] == queue['variant'])
    run = Path(queue['run_dir'])
    repo = Path(config['model_repo'])
    native = repo / 'inference_gt2d/scene_inference.py'
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location('fourdeval_elastic_native', native)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    save_name = variant['name'] + '_scannet_val88_gt2d_all'
    sys.argv = [str(native), queue['jobs'][0]['shard'], save_name, 'scannet',
                '--model-profile', variant['profile'], '--checkpoint', variant['checkpoint'],
                '--device', 'cuda', '--num-workers', '2', '--skip-data-validation',
                '--max-objects', '1000', '--object-filter-mode', 'gt2d_all',
                '--data-root', config['data_root'], '--output-root', str(run)]
    args = module.parse_args()
    install_large_interpolation()
    model = module.make_model(args)
    install_box_only_detector_fusion(model)
    install_tracker_mask_offload(model)
    original_dataset = module.make_dataset
    cache = {}
    owner = dict(pid=os.getpid(), start_tick=process_identity(os.getpid()),
                 gpu=options.gpu, log=options.log)
    while (job := claim_job(options.queue, owner)) is not None:
        print(f'[elastic-job] id={job["id"]} scene={job["scene"]} start={job["start"]} stop={job["stop"]}', flush=True)
        try:
            args.start_clip, args.max_clips = job['start'], job['stop']
            # Preserve the original shard's absolute indices and complete data
            # preprocessing. Only the execution interval changes.
            if job['shard'] not in cache:
                scenes = Path(job['shard']).read_text().split()
                frozen = queue.get('shard_manifests', {}).get(job['shard'])
                if frozen:
                    rows = json.loads((Path(frozen) / 'CA-1M.json').read_text())
                    if [Path(row[0]).stem for row in rows] != scenes:
                        raise ValueError('Frozen native manifest has different scene ordering')
                    # Reuse the exact native frame counts without reparsing all
                    # scene metadata or rewriting a shared temporary manifest.
                    module.make_manifest = lambda *a, **kw: frozen
                # Native make_manifest writes a shared filename before reading it.
                with (run / 'elastic/dataset.lock').open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    cache[job['shard']] = original_dataset(scenes, 'scannet', save_name, args)
            dataset = cache[job['shard']]
            for index in range(job['scene_start'], job['stop']):
                if Path(dataset.meta_file_paths[index][2]).stem != job['scene']:
                    raise ValueError('Scheduled clip interval crosses a scene boundary')
            if not job['scene_start'] <= job['start'] < job['stop'] <= len(dataset):
                raise ValueError('Invalid remaining clip interval')
            if (job['stop'] < len(dataset) and
                Path(dataset.meta_file_paths[job['stop']][2]).stem == job['scene']):
                raise ValueError('Scene job would mark an incomplete scene done')
            module.make_dataset = lambda *a, **kw: dataset
            # Passing only this scene also limits native final done flags when
            # the interval ends at the end of the original shard dataset.
            module.run_inference([job['scene']], save_name, 'scannet', model, args)
            pred = run / 'BoxDet/pred' / save_name / job['scene']
            pred.mkdir(parents=True, exist_ok=True)
            (pred / 'done.flag').write_text('done\n')
            finish_job(options.queue, job['id'], os.getpid(), phase='complete',
                       completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            print(f'[elastic-done] id={job["id"]}', flush=True)
        except Exception as exc:
            finish_job(options.queue, job['id'], os.getpid(), phase='failed', error=str(exc))
            traceback.print_exc()
            # Release CUDA allocations before the scheduler starts a new worker.
            raise
    print('[elastic-worker-done] no pending scenes', flush=True)


if __name__ == '__main__':
    main()
