"""Atomic scene-job ownership for local multi-GPU encoder workers."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from scripts.run_scannet_models import write_json


@contextmanager
def locked_queue(path):
    path = Path(path)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        queue = json.loads(path.read_text())
        yield queue
        write_json(path, queue)


def claim_job(path, owner):
    with locked_queue(path) as queue:
        pending = [job for job in queue['jobs'] if job['phase'] == 'pending']
        if not pending:
            return None
        job = max(pending, key=lambda x: x['stop'] - x['start'])
        job.update(phase='running', **owner)
        job['attempts'] = job.get('attempts', 0) + 1
        return dict(job)


def finish_job(path, job_id, pid, **values):
    with locked_queue(path) as queue:
        job = next(x for x in queue['jobs'] if x['id'] == job_id)
        if job.get('pid') != pid or job['phase'] != 'running':
            raise RuntimeError('Scene ownership changed while worker was running')
        job.update(values)
