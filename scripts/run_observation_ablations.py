"""Run matched RGB/box ablations only after native-answer QA passes its audit."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import urllib.request

from core.io import read_json, write_json
from core.response_constraints import VERSION, matches_constraint
from core.runner import output_lock
from scripts.finish_scannet_recovery import process_identity
from scripts.gpu_lease import acquire_gpu

ROOT=Path(__file__).resolve().parents[1]


def audit_answers(root):
    root=Path(root); config=read_json(root/'config.json'); reports=[]
    for job in config['jobs']:
        expected={r['sample_id'] for r in read_json(job['manifest'])['samples']}
        for variant in config['variants']:
            path=root/job['name']/variant['name']/'qa.json'
            result=read_json(path); rows=result['results']
            if (len(rows)!=len(expected) or {r['sample_id'] for r in rows}!=expected
                    or result['config'].get('answer_format')!=VERSION):
                raise ValueError(f'Incomplete or wrong-protocol answer audit: {path}')
            invalid=[r['sample_id'] for r in rows if r['status']!='ok'
                     or r.get('finish_reason') not in {'stop','eos_token'}
                     or not r.get('response_constraint')
                     or not matches_constraint(r['raw_output'],r['response_constraint'])]
            if invalid:
                raise ValueError(f'Answer audit failed for {path}: {len(invalid)} invalid/truncated rows')
            reports.append({'job':job['name'],'variant':variant['name'],'samples':len(rows),
                            'invalid':0,'score':result['primary_metric']})
    return {'passed':True,'reports':reports,'total_responses':sum(r['samples'] for r in reports)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-root',required=True)
    args=parser.parse_args()
    config=read_json(args.config); out=Path(args.output_root).resolve(); out.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'phase':'starting','tasks':{},'workers':{}}
    guard=threading.RLock(); children=[]
    def save(**fields):
        with guard:
            state.update(fields);state['updated']=time.time();write_json(out/'status.json',state)
    def task_status(key,**fields):
        with guard:
            state['tasks'].setdefault(key,{}).update(fields);save()
    def spawn(command,log,gpu):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                 OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('ab') as stream:
            child=subprocess.Popen(list(map(str,command)),cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        with guard:children.append(child)
        return child
    def stop(child):
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait()
    def interrupted(signum,frame):
        for child in children:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        os._exit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    def wait_controller(path,phase):
        save(phase=phase)
        deadline=time.monotonic()+72*3600
        while True:
            record=read_json(path)
            identity=record.get('start_time')
            if identity is None:
                raise ValueError(f'Prerequisite process has no recorded start time: {path}')
            if process_identity(record['pid'])!=identity:return
            if time.monotonic()>deadline:raise TimeoutError(f'Prerequisite did not finish: {path}')
            time.sleep(15)

    with output_lock(out/'status.json'):
        if (out/'config.json').exists() and read_json(out/'config.json')!=config:
            raise ValueError('Ablation configuration changed; use a new output root')
        write_json(out/'config.json',config)
        try:
            wait_controller(Path(config['answer_run'])/'process.json','waiting_for_answer_audit')
            audit=audit_answers(config['answer_run']);write_json(out/'answer_audit.json',audit)
            save(phase='answer_audit_passed',answer_audit=audit)
            wait_controller(config['source_process_file'],'waiting_for_source_evaluations')
            tasks=queue.Queue()
            for job in config['jobs']:
                for mode in job.get('modes',['rgb','boxes','rgb_boxes']):
                    variants=[{'name':'shared_rgb'}] if mode=='rgb' else job['geometries']
                    for variant in variants:
                        if mode!='rgb' and not Path(variant['path']).is_file():
                            raise FileNotFoundError(variant['path'])
                        key=f"{job['name']}/{variant['name']}/{mode}"
                        tasks.put((job,variant,mode,key));task_status(key,phase='queued')
            save(phase='running')

            def worker(gpu):
                server=None;lease=None
                try:
                    lease=acquire_gpu(gpu)
                    used=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
                    if any(int(m)>1024 for i,m in (s.split(',') for s in used.splitlines()) if int(i)==gpu):
                        raise RuntimeError(f'GPU {gpu} is occupied')
                    port=config.get('base_port',23500)+gpu
                    server=spawn([config['llm_python'],'-m','vllm.entrypoints.openai.api_server',
                        '--model',config['llm_model'],'--served-model-name',config['llm_name'],
                        '--host','127.0.0.1','--port',port,'--dtype','bfloat16','--max-model-len','131072',
                        '--gpu-memory-utilization','.82','--max-num-seqs','8','--generation-config','vllm',
                        '--no-enable-log-requests','--limit-mm-per-prompt','{"image":32,"video":0}',
                        '--mm-processor-kwargs',json.dumps({'min_pixels':65536,
                            'max_pixels':config.get('max_image_pixels',262144)})],
                        out/'logs'/f'server_gpu{gpu}.log',gpu)
                    deadline=time.monotonic()+1800
                    while True:
                        if server.poll() is not None or time.monotonic()>deadline:
                            raise RuntimeError(f'Multimodal server failed on GPU {gpu}')
                        try:
                            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models',timeout=3) as response:
                                if config['llm_name'] in [r['id'] for r in json.load(response)['data']]:break
                        except OSError:pass
                        time.sleep(3)
                    with guard:state['workers'][str(gpu)]={'phase':'running','server_pid':server.pid};save()
                    while True:
                        try:job,variant,mode,key=tasks.get_nowait()
                        except queue.Empty:break
                        try:
                            target=out/key/'qa.json';task_status(key,phase='running',gpu=gpu)
                            command=[sys.executable,'eval.py','--benchmark',job['benchmark'],'--data',job['data'],
                                '--split',job['split'],'--model',config['llm_name'],
                                '--base-url',f'http://127.0.0.1:{port}/v1','--temperature','0','--seed','0',
                                '--max-tokens','4096','--timeout','300','--concurrency','4','--batch-size','16',
                                '--answer-format','native','--observation-mode',mode,'--media-manifest',job['manifest'],
                                '--geometry-decimals','4','--max-prompt-chars','600000','--max-image-pixels',
                                config.get('max_image_pixels',262144),'--video-frames','16',
                                '--extra-body','{"chat_template_kwargs":{"enable_thinking":false}}',
                                '--run-label',key,'--output',target,'--resume']
                            if mode!='rgb':command+=['--geometry',variant['path']]
                            if job.get('dataset'):command+=['--dataset',job['dataset']]
                            child=spawn(command,out/key/'eval.log',gpu)
                            if child.wait()!=0:raise RuntimeError(f'Evaluation failed: {key}')
                            result=read_json(target)
                            expected=len(read_json(job['manifest'])['samples'])
                            if result['num_samples']!=expected or set(result['status_counts'])!={'ok'}:
                                raise ValueError('Ablation failed completeness/answer audit')
                            task_status(key,phase='complete',samples=expected,metric=result['primary_metric'],result=str(target))
                        except Exception as exc:task_status(key,phase='failed',error=f'{type(exc).__name__}: {exc}')
                        finally:tasks.task_done()
                    with guard:state['workers'][str(gpu)]['phase']='complete';save()
                except Exception as exc:
                    with guard:state['workers'][str(gpu)]={'phase':'failed','error':str(exc)};save()
                finally:
                    stop(server)
                    if lease:lease.close()

            with ThreadPoolExecutor(len(config['gpus'])) as pool:
                list(pool.map(worker,config['gpus']))
            rows=[]
            for key,result in state['tasks'].items():
                benchmark,variant,mode=key.split('/')
                rows.append({'benchmark':benchmark,'variant':variant,'mode':mode,
                    'phase':result['phase'],'samples':result.get('samples'),
                    'score_100':result.get('metric',{}).get('score_100'),
                    'result':result.get('result'),'error':result.get('error')})
            write_json(out/'summary.json',rows)
            if rows:
                with (out/'summary.csv').open('w') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            if any(r['phase']!='complete' for r in state['tasks'].values()):
                raise RuntimeError('Some ablations remain incomplete; inspect task and worker status')
            save(phase='complete')
        except Exception as exc:
            save(phase='failed',error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            for child in children:stop(child)


if __name__=='__main__':main()
