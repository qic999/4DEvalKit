"""Run V-STaR geometry and three observation modes using leased GPUs.

This controller can wait for scripts.download_vstar, then resume preparation,
geometry, QA and (when configured) the native 72B semantic judge. A real pilot
must finish successfully before starting a separate full run.
"""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from core.io import read_json, write_json
from core.runner import output_lock
from scripts.gpu_lease import acquire_gpu
from scripts.reasoner_pool import ReasonerPool, wait_for_free_gpu

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); c = read_json(a.config); out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = dict(phase='starting', pid=os.getpid(), modes={})
    pool, active, leases = None, None, []

    def update(**kw):
        state.update(kw, updated=time.time()); write_json(out/'status.json', state)

    def stop(signum=None, frame=None):
        if active and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            active.wait()
        if pool: pool.stop()
        for lease in leases: lease.close()
        if signum:
            update(phase='interrupted'); raise SystemExit(128+signum)

    def run(args, log, extra_env=None):
        nonlocal active
        with (out/log).open('ab') as stream:
            active = subprocess.Popen(list(map(str, args)), cwd=ROOT,
                env=dict(os.environ, PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
                         OPENBLAS_NUM_THREADS='2', **(extra_env or {})), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        if active.wait(): raise RuntimeError(f'Command failed; inspect {out/log}')
        active = None

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    with output_lock(out/'suite.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != c:
            raise ValueError('Changed configuration; choose a new output directory')
        write_json(out/'config.json', c)
        try:
            if c.get('download_status'):
                update(phase='waiting_for_videos')
                deadline = time.monotonic()+c.get('download_wait_seconds', 24*3600)
                while True:
                    status = read_json(c['download_status'])
                    if status['phase'] == 'complete': break
                    if status['phase'] == 'failed' or time.monotonic() > deadline:
                        raise RuntimeError('Dataset download failed or timed out')
                    time.sleep(20)
            if c.get('pilot_status'):
                pilot = read_json(c['pilot_status'])
                if pilot['phase'] != 'complete':
                    raise RuntimeError('Required pilot has not completed all inference and scoring')
            prepared = out/'prepared'
            args = [sys.executable, '-m', 'scripts.run_vstar', 'prepare', '--annotations', c['annotations'],
                    '--video-root', c['video_root'], '--output', prepared]
            if c.get('limit'): args += ['--limit', c['limit']]
            update(phase='preparing'); run(args, 'prepare.log')
            encoding = dict(c, modes=[], jobs=[dict(benchmark='V-STaR', prepared=str(prepared),
                                                   video_frames=c.get('video_frames', 16))])
            write_json(out/'encoding_config.json', encoding)
            update(phase='encoding')
            run([sys.executable, '-m', 'scripts.run_additional_4d', '--config', out/'encoding_config.json',
                 '--output', out/'encoding'], 'encoding.log')
            if read_json(out/'encoding/status.json')['phase'] != 'complete':
                raise RuntimeError('Geometry generation did not complete')
            pc = dict(c, gpus=c.get('reasoner_gpus', c['gpus']))
            pool = ReasonerPool(pc, out, lambda gpu, **kw: update(server_gpu=gpu, server_state=kw))
            update(phase='starting_reasoners'); endpoints = pool.start()
            for mode in c.get('modes', ['boxes', 'rgb', 'rgb_boxes']):
                update(phase='reasoning', current_mode=mode)
                args = [sys.executable, '-m', 'scripts.run_vstar', 'infer', '--prepared', prepared,
                    '--mode', mode, '--model', c['llm_name'], '--base-urls', *endpoints,
                    '--geometry', out/'encoding/V-STaR/geometry/geometry.json', '--output', out/'qa'/mode,
                    '--workers', c.get('qa_concurrency', 4), '--video-frames', c.get('video_frames', 16),
                    '--max-image-pixels', c.get('max_image_pixels', 262144), '--max-tokens', c.get('max_tokens', 8192),
                    '--extra-body', '{"chat_template_kwargs":{"enable_thinking":false}}']
                run(args, f'{mode}.log')
                state['modes'][mode] = 'inference_complete'; update()
            pool.stop(); pool = None
            if c.get('judge_model'):
                if c.get('judge_download_status'):
                    update(phase='waiting_for_judge_weights')
                    deadline = time.monotonic()+c.get('download_wait_seconds', 24*3600)
                    while True:
                        status = read_json(c['judge_download_status'])
                        if status['phase'] == 'complete': break
                        if status['phase'] == 'failed' or time.monotonic() > deadline:
                            raise RuntimeError('Judge download failed or timed out')
                        time.sleep(20)
                for gpu in sorted(c['judge_gpus']):
                    leases.append(acquire_gpu(gpu))
                    wait_for_free_gpu(gpu, lambda gpu, **kw: update(judge_gpu=gpu, judge_state=kw))
                for mode in state['modes']:
                    update(phase='scoring', current_mode=mode)
                    run([c['llm_python'], '-m', 'scripts.run_vstar', 'score', '--prepared', prepared,
                        '--predictions', out/'qa'/mode/'predictions.json', '--judge-model', c['judge_model'],
                        '--output', out/f'{mode}_metrics.json'], f'{mode}_judge.log',
                        {'CUDA_VISIBLE_DEVICES': ','.join(map(str, c['judge_gpus']))})
                    state['modes'][mode] = 'complete'; update()
                update(phase='complete')
            else:
                update(phase='inference_complete_scoring_pending')
            write_json(out/'suite.json', state)
        except Exception as exc:
            update(phase='failed', error=f'{type(exc).__name__}: {exc}'); raise
        finally:
            stop()


if __name__ == '__main__':
    main()
