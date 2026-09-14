"""Fresh Full/Small ScanNet val88 inference followed by Qwen reasoning.

Uses the existing model repository's validated GT-2D loader and all 88 scenes.
This is the ScanNet subset (2,071 questions), not all VSI sources or the core
static/dynamic suite. Never substitutes old predictions for model inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def file_info(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--postprocess-only', action='store_true',
                   help='Require all scene done flags; merge/score without launching encoders')
    p.add_argument('--plan', action='store_true')
    args = p.parse_args()
    if args.postprocess_only and not args.resume:
        p.error('--postprocess-only requires --resume')
    config = json.loads(Path(args.config).read_text())
    out = Path(args.output_root).resolve()
    repo = Path(config['model_repo']).resolve(strict=True)
    scene_list = repo / 'inference_gt2d/scannet_val88.txt'
    scenes = scene_list.read_text().split()
    qa_path = Path(config['qa_json']).resolve(strict=True)
    qa = json.loads(qa_path.read_text())
    selected = [q for q in qa if q.get('dataset') == 'scannet']
    if len(scenes) != 88 or len(set(scenes)) != 88 or len(selected) != 2071:
        raise ValueError('Expected the complete ScanNet val88 / 2,071 question subset')
    if {q['scene_name'] for q in selected} != set(scenes):
        raise ValueError('Scene list and QA scenes do not agree')
    variants = config['variants']
    gpu_ids = [str(g) for v in variants for g in v['gpus']] + [str(config['llm_gpu'])]
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError('GPU allocations must be disjoint')
    inputs = {v['name']: file_info(v['checkpoint']) for v in variants}
    for executable in ['encoder_python', 'llm_python', 'eval_python']:
        if not os.access(config[executable], os.X_OK):
            raise ValueError(f'Python executable unavailable: {config[executable]}')
    file_info(Path(config['llm_model']) / 'config.json')
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = dict(config=config, config_sha256=fingerprint, checkpoints=inputs,
                    scene_ids=scenes, question_count=len(selected),
                    qa_sha256=hashlib.sha256(qa_path.read_bytes()).hexdigest(),
                    toolkit_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    protocol='GT 2D boxes/categories + predicted 3D OBB; GT camera poses',
                    scope='VSI-Bench ScanNet val88 subset; all 88 scenes and 2071 questions per variant',
                    not_launched=['Remaining VSI sources', 'Other core benchmarks', 'Dynamic benchmarks'])
    if args.plan:
        print(json.dumps(manifest, indent=2))
        return
    if out.exists():
        if not args.resume:
            raise FileExistsError(f'{out} exists; use a fresh run directory or --resume')
        previous = json.loads((out / 'manifest.json').read_text())
        if previous['config_sha256'] != fingerprint or previous['checkpoints'] != inputs:
            raise ValueError('Cannot resume with changed settings/checkpoints')
        if (out / 'status.json').exists():
            previous_state = json.loads((out / 'status.json').read_text())
            try:
                os.kill(int(previous_state['pid']), 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError('Previous controller still exists; wait before resuming')
    else:
        out.mkdir(parents=True)
        manifest['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(out / 'manifest.json', manifest)
    if args.postprocess_only:
        for v in variants:
            pred = out / 'BoxDet/pred' / (v['name'] + '_scannet_val88_gt2d_all')
            if {x.parent.name for x in pred.glob('*/done.flag')} != set(scenes):
                raise RuntimeError(f'{v["name"]}: postprocessing requires all 88 completed scenes')
        gpu_ids = [str(config['llm_gpu'])]
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    for line in usage.splitlines():
        index, used = [s.strip() for s in line.split(',')]
        if index in gpu_ids and int(used) > int(config.get('max_existing_gpu_memory_mb', 1024)):
            raise RuntimeError(f'GPU {index} already uses {used} MiB; allocation no longer free')
    port = int(config['llm_port'])
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    lock = threading.Lock()
    server_lock = threading.Lock()
    state = {v['name']: {'phase': 'queued'} for v in variants}
    children = []
    server = None

    def status(name, **values):
        with lock:
            state[name].update(values, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            write_json(out / 'status.json', dict(pid=os.getpid(), variants=state))
        print(name, values, flush=True)

    def spawn(command, log, env, cwd):
        with log.open('a') as stream:
            process = subprocess.Popen(list(map(str, command)), cwd=cwd, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        with lock:
            children.append(process)
        return process

    def run(command, log, env=None, cwd=ROOT):
        process = spawn(command, log, env or os.environ.copy(), cwd)
        rc = process.wait()
        if rc:
            raise RuntimeError(f'Command failed ({rc}); see {log}')

    def ensure_server():
        nonlocal server
        with server_lock:
            if server is not None:
                if server.poll() is not None:
                    raise RuntimeError('Reasoning server exited; see logs/qwen_server.log')
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(config['llm_gpu']),
                       PYTHONNOUSERSITE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
            # Match the environment used by the previous Qwen3.5 val88 runs.
            site = subprocess.check_output([config['llm_python'], '-c',
                    'import site; print(site.getsitepackages()[0])'], text=True).strip()
            libdirs = list((Path(site) / 'nvidia').glob('*/lib')) + [Path(site) / 'torch/lib']
            env['LD_LIBRARY_PATH'] = ':'.join(map(str, libdirs)) + ':' + env.get('LD_LIBRARY_PATH', '')
            command = [config['llm_python'], '-m', 'vllm.entrypoints.openai.api_server',
                       '--model', config['llm_model'], '--served-model-name', config['llm_name'],
                       '--host', '127.0.0.1', '--port', port, '--dtype', 'bfloat16',
                       '--max-model-len', '16384', '--gpu-memory-utilization', '0.82',
                       '--max-num-seqs', '32', '--generation-config', 'vllm',
                       '--no-enable-log-requests', '--language-model-only']
            server = spawn(command, logs / 'qwen_server.log', env, ROOT)
            deadline = time.monotonic() + 1800
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError('Qwen startup failed; see logs/qwen_server.log')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                        names = [x['id'] for x in json.load(response)['data']]
                        if config['llm_name'] not in names:
                            raise ValueError('Unexpected served model')
                    print(f'Qwen ready on port {port}; PID {server.pid}', flush=True)
                    return
                except OSError:
                    time.sleep(3)
            raise TimeoutError('Qwen startup exceeded 30 minutes')

    def evaluate(v):
        name = v['name']
        try:
            save_name = name + '_scannet_val88_gt2d_all'
            existing = out / 'qa' / (name + '.json')
            if args.postprocess_only and existing.is_file():
                result = json.loads(existing.read_text())
                if (result.get('num_samples') == len(selected)
                    and result.get('status_counts') == {'ok': len(selected)}
                    and len(result['results']) == len(selected)
                    and {str(x['source_id']) for x in result['results']}
                        == {str(x['id']) for x in selected}):
                    status(name, phase='complete', result=str(existing), completed_scenes=88)
                    return
            env = dict(os.environ, BOX_DATA_PATH=config['data_root'], BOX_OUTPUT_PATH=str(out),
                       PYTHON_BIN=config['encoder_python'], GPU_LIST=','.join(map(str, v['gpus'])),
                       SCENE_LIST=str(scene_list), RESUME='1' if args.resume else '0',
                       OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                       PYTHONUNBUFFERED='1', NUM_WORKERS='2', MAX_OBJECTS='1000',
                       OBJECT_FILTER_MODE='gt2d_all')
            if config.get('encoder_base_python'):
                env['FOURDEVAL_ENCODER_PYTHON'] = config['encoder_base_python']
            env['FOURDEVAL_BOX_ONLY_FUSION'] = '1' if config.get('box_only_detector_fusion') else '0'
            env['SKIP_DATA_VALIDATION'] = '1' if config.get('data_prevalidated') else '0'
            status(name, phase='3d_metrics' if args.postprocess_only else 'encoder_and_3d_metrics', gpus=v['gpus'],
                   log=str(logs / (name + '_encoder.log')), checkpoint=v['checkpoint'])
            if args.postprocess_only:
                exp_name = 'merged_bbox_3dconf_top5meanwhl1r_timestamp'
                for scene in scenes:
                    run([config['encoder_python'], repo / 'inference_gt2d/merge_json.py',
                         '--scene_id', scene, '--pred_dir', save_name, '--exp_name', exp_name],
                        logs / (name + '_merge_recovery.log'), env, repo)
                run([config['encoder_python'], repo / 'inference_gt2d/compute_metrics_3d.py',
                     'scannet', '--pred-dir', save_name, '--exp-name', exp_name],
                    out / 'runs' / save_name / 'metrics.log', env, repo)
            else:
                run(['bash', repo / 'inference_gt2d/run_scannet_fast16.sh',
                     v['profile'], v['checkpoint'], save_name], logs / (name + '_encoder.log'), env, repo)
            pred = out / 'BoxDet/pred' / save_name
            if {x.parent.name for x in pred.glob('*/done.flag')} != set(scenes):
                raise RuntimeError('Encoder did not mark all 88 scenes complete')
            status(name, phase='convert_geometry')
            geometry = out / 'geometry' / (name + '.json')
            if not geometry.exists():
                run([config['eval_python'], '-m', 'scripts.convert_geometry', '--format', 'merged',
                     '--input', pred, '--output', geometry,
                     '--merged-name', 'merged_bbox_3dconf_top5meanwhl1r_timestamp.json',
                     '--quaternion-order', 'xyzw', '--units', 'm', '--coordinate-frame', 'scannet_world_z_up',
                     '--geometry-source', 'predicted', '--encoder', v['encoder'],
                     '--checkpoint', v['checkpoint'], '--proposal-source', 'gt_2d',
                     '--label-source', 'gt_category', '--camera-pose-source', 'ground_truth'],
                    logs / (name + '_convert.log'))
            status(name, phase='wait_reasoning_server', geometry=str(geometry))
            ensure_server()
            result = out / 'qa' / (name + '.json')
            status(name, phase='reasoning', result=str(result), total_questions=len(selected))
            run([config['eval_python'], 'eval.py', '--benchmark', 'VSI-Bench', '--data', qa_path,
                 '--dataset', 'scannet', '--geometry', geometry, '--model', config['llm_name'],
                 '--base-url', f'http://127.0.0.1:{port}/v1', '--temperature', '0', '--seed', '0',
                 '--max-tokens', '512', '--concurrency', '8', '--batch-size', '16',
                 '--extra-body', '{"chat_template_kwargs":{"enable_thinking":false}}',
                 '--run-label', name + '_gt2d_pred3d_gtpose', '--output', result,
                 '--save-prompts', '--resume', '--verbose'], logs / (name + '_qa.log'))
            status(name, phase='complete', result=str(result), completed_scenes=88)
        except Exception as exc:
            status(name, phase='failed', error=str(exc))

    def cleanup():
        # Only processes owned by this run are signalled, never other GPU jobs.
        for proc in children:
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    def interrupted(signum, frame):
        cleanup()
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with ThreadPoolExecutor(max_workers=len(variants)) as pool:
            futures = [pool.submit(evaluate, v) for v in variants]
            while not all(f.done() for f in futures):
                for v in variants:
                    name = v['name']
                    pred = out / 'BoxDet/pred' / (name + '_scannet_val88_gt2d_all')
                    count = sum(1 for _ in pred.glob('*/done.flag'))
                    with lock:
                        state[name]['completed_scenes'] = count
                        write_json(out / 'status.json', dict(pid=os.getpid(), variants=state))
                print('progress', {k: (v['phase'], v.get('completed_scenes', 0)) for k,v in state.items()}, flush=True)
                time.sleep(30)
            for future in futures:
                future.result()
    finally:
        cleanup()
    if any(v['phase'] != 'complete' for v in state.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
