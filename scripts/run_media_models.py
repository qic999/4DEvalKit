"""Background RGB -> proposals -> Full/Small geometry -> text QA pipeline.

GPU 0 produces proposals, GPUs 1-6 encode disjoint question shards, and GPU 7
serves reasoning after the earlier ScanNet run releases it. Every phase records
its own status. Failed jobs remain failed and never become aggregate scores.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from core.geometry import GeometryStore
from core.io import digest, read_json, write_json
from core.runner import output_lock
from scripts.finish_scannet_recovery import process_identity

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-root', required=True)
    args = p.parse_args()
    config = read_json(args.config)
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    jobs = config['jobs']
    states = {j['name']: {'phase':'queued'} for j in jobs}
    lock = threading.RLock()
    children = []
    proposal_ready = {j['name']: threading.Event() for j in jobs}
    geometry_ready = {j['name']: threading.Event() for j in jobs}
    def status(name, **fields):
        with lock:
            states[name].update(fields)
            write_json(out/'status.json', {'pid':os.getpid(), 'jobs':states, 'updated':time.time()})
    def spawn(cmd, log, gpu=None):
        log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                   PYTHONUNBUFFERED='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        with log.open('ab') as stream:
            proc = subprocess.Popen(list(map(str,cmd)), cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        with lock:
            children.append(proc)
        return proc
    def cleanup(*_):
        for proc in children:
            if proc.poll() is None:
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass
    def interrupted(signum, frame):
        cleanup()
        os._exit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def produce():
        for job in jobs:
            name = job['name']
            try:
                manifest = Path(job['manifest'])
                deadline = time.monotonic()+24*3600
                while not manifest.exists():
                    if time.monotonic()>deadline: raise TimeoutError('Media manifest was not prepared')
                    time.sleep(5)
                dest = out/name/'proposals'
                status(name, proposal_phase='running')
                cmd = [config['llm_python'], '-u','-m','scripts.generate_media_proposals',
                       '--manifest', manifest, '--output', dest, '--model', config['detector_model'],
                       '--max-objects',str(config.get('max_objects',40))]
                proc = spawn(cmd,out/name/'logs/proposals.log',config['proposal_gpu'])
                status(name, proposal_pid=proc.pid)
                while not (dest/'index.json').exists() and proc.poll() is None:
                    time.sleep(2)
                proposal_ready[name].set()
                rc = proc.wait()
                if rc: raise RuntimeError(f'Proposal generation exited {rc}')
                status(name, proposal_phase='complete')
            except Exception as exc:
                status(name, proposal_phase='failed', proposal_error=str(exc))
                proposal_ready[name].set()

    def encode():
        for job in jobs:
            name = job['name']
            workers = []
            try:
                proposal_ready[name].wait()
                index_file = out/name/'proposals/index.json'
                if not index_file.exists() or states[name].get('proposal_phase')=='failed':
                    raise RuntimeError('Proposal generation failed')
                status(name, phase='encoding')
                for variant in config['variants']:
                    for shard,gpu in enumerate(variant['gpus']):
                        dest = out/name/variant['name']/f'shard_{shard}'
                        cmd = [config['encoder_python'],'-u','-m','scripts.encode_media_geometry',
                               '--manifest',job['manifest'],'--proposals',out/name/'proposals',
                               '--model-repo',config['model_repo'],'--checkpoint',variant['checkpoint'],
                               '--profile',variant['profile'],'--output',dest,'--shard-index',shard,
                               '--num-shards',len(variant['gpus']),'--wait-inputs']
                        workers.append(spawn(cmd,out/name/'logs'/f"{variant['name']}_{shard}.log",gpu))
                status(name, encoder_pids=[p.pid for p in workers])
                while any(p.poll() is None for p in workers):
                    if any(p.poll() not in {None,0} for p in workers):
                        raise RuntimeError('An encoder shard failed; inspect per-shard logs')
                    time.sleep(5)
                expected = set(read_json(index_file)['config']['sample_ids'])
                coverage = {}
                for variant in config['variants']:
                    scenes = {}
                    for shard in range(len(variant['gpus'])):
                        path = out/name/variant['name']/f'shard_{shard}'/'geometry.json'
                        items = GeometryStore(path).scenes
                        if scenes.keys() & items.keys(): raise ValueError('Duplicate geometry IDs across shards')
                        scenes.update(items)
                    if set(scenes)!=expected: raise ValueError('Incomplete geometry coverage')
                    target = out/name/variant['name']/'geometry.json'
                    write_json(target, {'scenes':scenes})
                    coverage[variant['name']] = {'samples':len(scenes),
                        'empty_scenes':sum(not (s.get('objects') or s.get('tracks') or any(v['objects'] for v in s.get('views',[]))) for s in scenes.values())}
                status(name, phase='geometry_complete', coverage=coverage)
            except Exception as exc:
                for proc in workers:
                    if proc.poll() is None: os.killpg(proc.pid,signal.SIGTERM)
                status(name, phase='failed', error=str(exc))
            finally:
                geometry_ready[name].set()

    def reason():
        server = None
        try:
            for job in jobs:
                name = job['name']
                geometry_ready[name].wait()
                if states[name]['phase']=='failed': continue
                if server is None:
                    prior = config.get('wait_for_scannet_status')
                    while prior and Path(prior).exists():
                        previous = read_json(prior)
                        if previous['phase'] in {'complete','failed'} or process_identity(previous['pid']) is None: break
                        time.sleep(10)
                    used = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    if any(int(m)>1024 for i,m in (line.split(',') for line in used.splitlines()) if int(i)==config['llm_gpu']):
                        raise RuntimeError('Reasoning GPU is still occupied')
                    server = spawn([config['llm_python'],'-m','vllm.entrypoints.openai.api_server',
                        '--model',config['llm_model'],'--served-model-name',config['llm_name'],
                        '--host','127.0.0.1','--port',config['llm_port'],'--dtype','bfloat16','--max-model-len','131072',
                        '--gpu-memory-utilization','.82','--max-num-seqs','32','--generation-config','vllm',
                        '--no-enable-log-requests','--language-model-only'],out/'reasoning_server.log',config['llm_gpu'])
                    deadline = time.monotonic()+1800
                    while True:
                        if server.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Reasoning server startup failed')
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{config['llm_port']}/v1/models",timeout=3) as response:
                                if config['llm_name'] not in [x['id'] for x in __import__('json').load(response)['data']]:
                                    raise ValueError('Wrong reasoning model')
                            break
                        except OSError: time.sleep(3)
                try:
                    status(name, phase='reasoning')
                    results = []
                    for variant in config['variants']:
                        result = out/name/variant['name']/'qa.json'
                        cmd = [sys.executable,'eval.py','--benchmark',job['benchmark'],'--data',job['data'],
                            '--split',job['split'],'--geometry',out/name/variant['name']/'geometry.json',
                            '--model',config['llm_name'],'--base-url',f"http://127.0.0.1:{config['llm_port']}/v1",
                            '--temperature','0','--seed','0','--max-tokens','512','--concurrency','8','--batch-size','16',
                            '--geometry-decimals','4','--max-prompt-chars','600000',
                            '--extra-body','{"chat_template_kwargs":{"enable_thinking":false}}',
                            '--run-label',variant['name']+'_pred2d_pred3d_predpose','--output',result,'--resume']
                        proc = spawn(cmd,out/name/'logs'/f"{variant['name']}_qa.log")
                        if proc.wait(): raise RuntimeError('Reasoning job failed')
                        data = read_json(result)
                        expected = states[name]['coverage'][variant['name']]['samples']
                        if data['num_samples']!=expected or set(data['status_counts'])-{'ok','invalid_completion'}:
                            raise ValueError('Incomplete/request-failed QA results')
                        results.append({'variant':variant['name'], 'primary_metric':data['primary_metric'],
                                        'status_counts':data['status_counts'], 'result':str(result)})
                    status(name, phase='complete', results=results)
                except Exception as exc:
                    status(name, phase='failed', error=str(exc))
        finally:
            if server and server.poll() is None: os.killpg(server.pid,signal.SIGTERM)

    with output_lock(out/'status.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != config:
            raise ValueError('Run settings changed; choose a fresh output root')
        write_json(out/'config.json',config)
        try:
            with ThreadPoolExecutor(3) as pool:
                futures = [pool.submit(fn) for fn in [produce,encode,reason]]
                for future in futures: future.result()
        finally:
            cleanup()
    if any(s['phase']!='complete' for s in states.values()): raise SystemExit(1)


if __name__ == '__main__':
    main()
