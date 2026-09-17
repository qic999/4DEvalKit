"""Managed identical multimodal replicas, each holding a toolkit GPU lease."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import urllib.request

from scripts.gpu_lease import acquire_gpu

ROOT = Path(__file__).resolve().parents[1]


def wait_for_free_gpu(gpu, update, timeout=72*3600):
    deadline = time.monotonic() + timeout
    while True:
        used = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                        '--format=csv,noheader,nounits'], text=True)
        memory = next(int(m) for i, m in (s.split(',') for s in used.splitlines()) if int(i) == gpu)
        if memory <= 1024:
            return
        update(gpu, phase='waiting_for_available_memory', memory_used_mib=memory)
        if time.monotonic() > deadline:
            raise TimeoutError(f'GPU {gpu} remained occupied for {timeout} seconds')
        time.sleep(20)


class ReasonerPool:
    def __init__(self, config, output, update):
        self.config, self.output, self.update = config, Path(output), update
        self.servers, self.leases = {}, {}
        self.lock = threading.Lock()
        self.urls = [f"http://127.0.0.1:{config.get('base_port', 23600)+gpu}/v1"
                     for gpu in config['gpus']]

    def start(self):
        if not self.config['gpus'] or len(set(self.config['gpus'])) != len(self.config['gpus']):
            raise ValueError('Configure at least one unique GPU')
        with ThreadPoolExecutor(len(self.config['gpus'])) as workers:
            results = list(workers.map(self._start_one, self.config['gpus']))
        errors = [r for r in results if r]
        if errors:
            raise RuntimeError('; '.join(errors))
        return self.urls

    def _start_one(self, gpu):
        try:
            self.update(gpu, phase='waiting_for_gpu')
            lease = acquire_gpu(gpu)
            with self.lock:
                self.leases[gpu] = lease
            wait_for_free_gpu(gpu, self.update, self.config.get('gpu_wait_seconds', 72*3600))
            c = self.config; port = c.get('base_port', 23600) + gpu
            command = [c['llm_python'], '-m', 'vllm.entrypoints.openai.api_server',
                       '--model', c['llm_model'], '--served-model-name', c['llm_name'],
                       '--host', '127.0.0.1', '--port', str(port), '--dtype', 'bfloat16',
                       '--max-model-len', '131072', '--gpu-memory-utilization', '.82',
                       '--max-num-seqs', '8', '--generation-config', 'vllm',
                       '--no-enable-log-requests', '--limit-mm-per-prompt', '{"image":32,"video":0}',
                       '--mm-processor-kwargs', json.dumps({'min_pixels': 65536,
                           'max_pixels': c.get('max_image_pixels', 262144)})]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
            log = self.output / 'logs' / f'server_gpu{gpu}.log'; log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('ab') as stream:
                server = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            with self.lock:
                self.servers[gpu] = server
            self.update(gpu, phase='starting_server', pid=server.pid)
            deadline = time.monotonic() + 1800
            while True:
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f'Server startup failed on GPU {gpu}; inspect {log}')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as response:
                        if c['llm_name'] in [r['id'] for r in json.load(response)['data']]:
                            break
                except OSError:
                    pass
                time.sleep(3)
            self.update(gpu, phase='ready', pid=server.pid)
        except Exception as exc:
            self.update(gpu, phase='failed', error=str(exc))
            return str(exc)

    def stop(self):
        for server in list(self.servers.values()):
            if server.poll() is None:
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 45
        for server in list(self.servers.values()):
            try:
                server.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL); server.wait()
        for lease in list(self.leases.values()):
            lease.close()
        self.leases.clear()
