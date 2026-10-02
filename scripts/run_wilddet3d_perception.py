"""Persistent GPU queue for box-prompt perception, followed by official scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from core.io import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    args = p.parse_args()
    cfg = read_json(args.config)
    root = Path(cfg['output']).resolve()
    status_path = root/cfg.get('status_file', 'status.json')
    (root/'logs').mkdir(parents=True, exist_ok=True)
    jobs = []
    for dataset in cfg['datasets']:
        for model in cfg['models']:
            for shard in range(cfg['num_shards']):
                jobs.append({'dataset': dataset, 'model': model, 'shard': shard, 'state': 'pending'})
    active, scoring, scored = {}, {}, set()
    for dataset in cfg['datasets']:
        for model in cfg['models']:
            key = dataset['name']+'/'+model['name']
            status = root/key/'scoring_status.json'
            if status.exists() and read_json(status).get('complete'):
                scored.add(key)
    started = time.time()
    while True:
        for gpu, (proc, job, log) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            job.update(state='complete' if rc == 0 else 'failed', returncode=rc, finished_at=time.time())
            log.close()
            del active[gpu]
        for key, (proc, log) in list(scoring.items()):
            rc = proc.poll()
            if rc is not None:
                scored.add(key)
                log.close()
                del scoring[key]
                write_json(root/key/'scoring_status.json', {'complete': rc == 0, 'returncode': rc})
        gpu_memory = None
        if 'max_existing_gpu_memory_mb' in cfg:
            usage = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            gpu_memory = {int(row.split(',')[0]): int(row.split(',')[1]) for row in usage.strip().splitlines()}
        for gpu in cfg['gpus']:
            if gpu in active:
                continue
            if gpu_memory is not None and gpu_memory[gpu] > cfg['max_existing_gpu_memory_mb']:
                continue
            ready = next((j for j in jobs if j['state'] == 'pending' and Path(j['dataset']['manifest']).exists()), None)
            if ready is None:
                continue
            model, dataset = ready['model'], ready['dataset']
            key = dataset['name']+'/'+model['name']
            out = root/key
            done = out/f'shard_{ready["shard"]}.json'
            if done.exists() and read_json(done).get('complete'):
                ready['state'] = 'complete'
                continue
            cmd = [cfg['encoder_python'], '-u', '-m', 'scripts.encode_wilddet3d_perception',
                   '--manifest', dataset['manifest'], '--output', str(out),
                   '--shard-index', str(ready['shard']), '--num-shards', str(cfg['num_shards'])]
            for k in ['model-repo', 'checkpoint', 'profile', 'resolution', 'spatial-resolution', 'model-image-size']:
                cmd.extend(['--'+k, str(model[k])])
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            log = open(root/'logs'/f'{dataset["name"]}_{model["name"]}_{ready["shard"]}.log', 'a')
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            ready.update(state='running', pid=proc.pid, gpu=gpu, started_at=time.time())
            active[gpu] = (proc, ready, log)
        for dataset in cfg['datasets']:
            for model in cfg['models']:
                key = dataset['name']+'/'+model['name']
                group = [j for j in jobs if j['dataset']['name'] == dataset['name'] and j['model']['name'] == model['name']]
                if key in scored or key in scoring or not all(j['state'] == 'complete' for j in group):
                    continue
                if len(scoring) >= 2:
                    continue
                metrics = dataset.get('metrics', ['dist', 'bbox'])
                cmd = [cfg['scorer_python'], '-u', '-m', 'scripts.score_wilddet3d_perception',
                       '--upstream', cfg['upstream'], '--manifest', dataset['manifest'],
                       '--predictions', str(root/key), '--output', str(root/key/'metrics.json'), '--metrics', *metrics]
                log = open(root/'logs'/f'{dataset["name"]}_{model["name"]}_score.log', 'a')
                proc = subprocess.Popen(cmd, env=dict(os.environ, OMP_NUM_THREADS='2'),
                                        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                scoring[key] = (proc, log)
        write_json(status_path, {'pid': os.getpid(), 'started_at': started, 'updated_at': time.time(),
                   'jobs': [{k:v for k,v in j.items() if k not in ['dataset','model']} |
                            {'dataset':j['dataset']['name'], 'model':j['model']['name']} for j in jobs],
                   'scoring': list(scoring), 'scored': sorted(scored)})
        terminal = all(j['state'] in ['complete', 'failed'] for j in jobs)
        eligible_scores = sum(all(j['state']=='complete' for j in jobs if j['dataset']['name']==d['name'] and j['model']['name']==m['name'])
                              for d in cfg['datasets'] for m in cfg['models'])
        if terminal and not active and not scoring and len(scored) == eligible_scores:
            break
        if time.time()-started > cfg.get('deadline_hours', 72)*3600 and not active and not scoring:
            raise TimeoutError('Data preparation did not produce all manifests before deadline')
        time.sleep(10)


if __name__ == '__main__':
    main()
