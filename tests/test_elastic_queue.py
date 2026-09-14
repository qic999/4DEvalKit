from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from scripts.elastic_queue import claim_job, finish_job, locked_queue


def test_concurrent_workers_claim_each_scene_once(tmp_path):
    path = tmp_path / 'queue.json'
    jobs = [{'id': str(i), 'start': 1000 + i * 16, 'stop': 1016 + i * 16,
             'phase': 'pending'} for i in range(64)]
    path.write_text(json.dumps({'jobs': jobs}))

    def worker(pid):
        claimed = []
        while (job := claim_job(path, {'pid': pid})) is not None:
            claimed.append(job['id'])
            finish_job(path, job['id'], pid, phase='complete')
        return claimed

    with ThreadPoolExecutor(max_workers=8) as pool:
        claimed = [job for group in pool.map(worker, range(1, 9)) for job in group]
    assert len(claimed) == len(set(claimed)) == 64
    final = json.loads(path.read_text())['jobs']
    assert all(j['phase'] == 'complete' and j['attempts'] == 1 for j in final)
    assert [(j['start'], j['stop']) for j in final] == [(j['start'], j['stop']) for j in jobs]


def test_other_worker_cannot_complete_owned_scene(tmp_path):
    path = tmp_path / 'queue.json'
    path.write_text(json.dumps({'jobs': [{'id': 'scene', 'start': 7, 'stop': 12, 'phase': 'pending'}]}))
    claim_job(path, {'pid': 100})
    with pytest.raises(RuntimeError, match='ownership'):
        finish_job(path, 'scene', 200, phase='complete')
    assert json.loads(path.read_text())['jobs'][0]['phase'] == 'running'


def test_failed_queue_update_preserves_original(tmp_path):
    path = tmp_path / 'queue.json'
    path.write_text('{"jobs": []}')
    with pytest.raises(ValueError):
        with locked_queue(path) as queue:
            queue['jobs'].append({'id': 'uncommitted'})
            raise ValueError('aborted')
    assert json.loads(path.read_text()) == {'jobs': []}
