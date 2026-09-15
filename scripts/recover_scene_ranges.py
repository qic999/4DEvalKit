"""Finish disjoint native clip ranges, audit coverage, then mark one scene done."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time

from core.io import read_json, write_json
from core.runner import output_lock


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--template', required=True, help='Recorded native command JSON')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--start', type=int, required=True)
    p.add_argument('--stop', type=int, required=True)
    p.add_argument('--scene', required=True)
    p.add_argument('--save-name', required=True)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6')
    args = p.parse_args()
    run = Path(args.run_dir).resolve()
    out = run / 'recovery' / f'ranges_{args.start}_{args.stop}'
    out.mkdir(parents=True, exist_ok=True)
    state = {'pid': os.getpid(), 'phase': 'encoding', 'workers': []}
    children = []
    try:
        with output_lock(out / 'status.json'):
            original = read_json(args.template)['command']
            base = original[:]
            for flag in ['--max-clips', '--start-clip']:
                if flag in base:
                    i = base.index(flag)
                    del base[i:i+2]
            if '--skip-done' in base:
                base.remove('--skip-done')
            base[base.index('--output-root')+1] = str(run)
            gpus = args.gpus.split(',')
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
            for line in used.splitlines():
                gpu, mb = line.split(',')
                if gpu.strip() in gpus and int(mb) > 1024:
                    raise RuntimeError(f'GPU {gpu} is occupied')
            env = dict(os.environ, FOURDEVAL_ENCODER_PYTHON='/mnt/realccvl15/qchen76/env/sam3/bin/python',
                FOURDEVAL_BOX_ONLY_FUSION='1', FOURDEVAL_TRACKER_MASK_OFFLOAD='1', FOURDEVAL_LARGE_INTERPOLATION='1',
                FOURDEVAL_CHUNKED_ROPE='1', FOURDEVAL_INPLACE_ROPE='1', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONUNBUFFERED='1')
            for i,gpu in enumerate(gpus):
                start = args.start + (args.stop-args.start)*i//len(gpus)
                stop = args.start + (args.stop-args.start)*(i+1)//len(gpus)
                if start == stop:
                    continue
                command = base + ['--start-clip', str(start), '--max-clips', str(stop)]
                log = out / f'gpu_{gpu}.log'
                with log.open('xb') as stream:
                    proc = subprocess.Popen(command, env={**env, 'CUDA_VISIBLE_DEVICES': gpu},
                        stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT)
                children.append(proc)
                state['workers'].append({'pid': proc.pid, 'gpu': gpu, 'start': start, 'stop': stop, 'log': str(log)})
            while any(proc.poll() is None for proc in children):
                for proc, worker in zip(children, state['workers']):
                    worker['returncode'] = proc.poll()
                write_json(out / 'status.json', state)
                time.sleep(10)
            for proc, worker in zip(children, state['workers']):
                if proc.returncode != 0:
                    raise RuntimeError(f"Worker failed: {worker['log']}")
                text = Path(worker['log']).read_text()
                writes = re.findall(r'\[write\] scene=(\S+) clip=(\d+)', text)
                if any(scene != args.scene for scene, _ in writes):
                    raise RuntimeError('Clip range crossed another scene')
                covered = {int(n) for _,n in writes}
                covered.update(map(int, re.findall(r'\[skip\] empty datapoint at clip (\d+)', text)))
                if covered != set(range(worker['start'], worker['stop'])):
                    raise RuntimeError(f"Incomplete clip coverage: {worker['log']}")
                worker['returncode'] = 0
            done = run / 'BoxDet/pred' / args.save_name / args.scene / 'done.flag'
            done.write_text('done\n')
            state['phase'] = 'complete'
            write_json(out / 'status.json', state)
    except BaseException as exc:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        state.update(phase='failed', error=str(exc))
        write_json(out / 'status.json', state)
        raise


if __name__ == '__main__':
    main()
