import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.vstar import task_prompts, parse_prediction, summarize, TASKS, judge_namespace
from scripts.run_vstar import prepare, infer


def annotation():
    return dict(vid='v', question='What moves?', answer='SECRET_ANSWER', chain='SECRET_CHAIN',
                temporal_question='When does the red object move?',
                spatial_question='Where is the red object between 1 and 2 seconds?',
                spatial_question_2='Where is the red object?', timestamps=[1, 2],
                width=640, height=360, frame_count=90, fps=30, domain='Objects',
                bboxes=[dict(timestamp=1, xmin=17, ymin=19, xmax=27, ymax=29)])


def test_conditioned_prompts_keep_conditions_in_their_official_tasks():
    row = annotation()
    tasks = task_prompts(row)
    assert set(tasks) == set(TASKS)
    assert 'SECRET' not in json.dumps(tasks)
    changed = copy.deepcopy(row)
    changed['bboxes'][0]['xmin'] = 123
    altered = task_prompts(changed)
    assert [k for k in TASKS if tasks[k] != altered[k]] == ['answer_temporal_2']
    changed = copy.deepcopy(row)
    changed['timestamps'] = [0, 2]
    altered = task_prompts(changed)
    assert [k for k in TASKS if tasks[k] != altered[k]] == ['answer_spatial']


def test_prepare_separates_gt_and_never_uses_gt_temporal_crops(tmp_path):
    (tmp_path/'v.mp4').touch()
    ann = tmp_path/'annotations.json'
    ann.write_text(json.dumps([annotation(), annotation()]))
    out = tmp_path/'prepared'
    prepare(SimpleNamespace(annotations=str(ann), video_root=str(tmp_path), output=str(out), limit=None))
    manifest = json.loads((out/'manifest.json').read_text())
    assert len(manifest['samples']) == 2
    for row in manifest['samples']:
        assert row['input_metadata'] == {'scene': 'v'}
        assert 'timestamps' not in row
    assert 'SECRET' not in (out/'manifest.json').read_text()
    assert 'SECRET' not in (out/'tasks.json').read_text()
    assert 'SECRET_ANSWER' in (out/'scoring.json').read_text()
    (tmp_path/'v.mp4').unlink()
    with pytest.raises(ValueError, match='0 matches'):
        prepare(SimpleNamespace(annotations=str(ann), video_root=str(tmp_path), output=str(out), limit=None))


@pytest.mark.parametrize('raw', ['[2,1]', '[NaN,2]', '[true,2]', 'explanation [1,2]', '[1]'])
def test_reject_malformed_temporal(raw):
    assert parse_prediction('answer_temporal', raw) is None


def test_parse_original_pixel_boxes_and_intervals():
    assert parse_prediction('answer_temporal', '```json\n[1,2]\n```') == [1, 2]
    assert parse_prediction('answer_spatial', '{"2":[1,2,640,360]}') == {'2': [1, 2, 640, 360]}
    assert parse_prediction('answer_spatial', '{"2.5":[1,2,3,4]}') is None
    assert parse_prediction('answer_spatial', '{"2":[3,2,1,4]}') is None
    assert parse_prediction('answer_spatial', '{"2":[0,0,NaN,4]}') is None


def test_joint_metrics_denominators_and_perfect_limit():
    perfect = dict(rating=3, chains={k: dict(temporal_iou=1, spatial_iou=1, spatial_ap=[1]*5)
                                    for k in ('1', '2')})
    missing = dict(rating=-1, chains={k: dict(temporal_iou=0, spatial_iou=0, spatial_ap=[0]*5)
                                     for k in ('1', '2')})
    stats = summarize([perfect, missing])
    assert stats['vqa_accuracy'] == stats['mAM'] == .5
    assert stats['chains']['1']['joint_all'] == .5
    assert stats['invalid_judge_outputs'] == 1
    assert stats['mLGM'] == pytest.approx(__import__('math').log(2))
    stats = summarize([perfect])
    assert stats['mLGM'] is None and stats['chains']['1']['LGM_infinite']
    json.dumps(stats, allow_nan=False)


def test_native_judge_loader_does_not_execute_upstream_top_level(tmp_path):
    path = tmp_path/'judge.py'
    path.write_text('raise RuntimeError("must not initialize upstream model")\n'
                    'system_prompt = "prompt"\ntmpl = "template"\n'
                    'def qwen2_5_evaluation(question, gt, candidate):\n'
                    '    return model(question, gt, candidate, system_prompt, tmpl)\n')
    ns = judge_namespace(path, lambda *a: a, None)
    assert ns['qwen2_5_evaluation']('q', 'gt', 'pred') == ('q', 'gt', 'pred', 'prompt', 'template')


def test_box_inference_never_opens_labels_and_resumes_five_subtasks(tmp_path, monkeypatch):
    (tmp_path/'v.mp4').write_bytes(b'test media signature')
    ann = tmp_path/'annotation.json'; ann.write_text(json.dumps([annotation()]))
    prepared = tmp_path/'prepared'
    prepare(SimpleNamespace(annotations=str(ann), video_root=str(tmp_path), output=str(prepared), limit=None))
    (prepared/'scoring.json').unlink()  # The answerer must not need this file.
    geometry = tmp_path/'geometry.json'
    geometry.write_text(json.dumps({'vstar:0': {'units': 'm', 'coordinate_frame': 'camera', 'objects': []}}))
    calls = []

    class Engine:
        def __init__(self, **kwargs): pass
        def infer(self, messages):
            calls.append(messages)
            prompt = json.dumps(messages)
            assert 'SECRET' not in prompt
            text = '[0,2]' if 'start_seconds' in prompt else '{}' if 'JSON object' in prompt else 'red object'
            return dict(raw_output=text, status='ok', finish_reason='stop')

    monkeypatch.setattr('core.inference.APIInferenceEngine', Engine)
    a = SimpleNamespace(prepared=str(prepared), geometry=str(geometry), mode='boxes', model='fake-for-test',
        base_urls=['http://127.0.0.1:1/v1'], video_frames=16, max_image_pixels=262144,
        max_tokens=8192, extra_body='{}', output=str(tmp_path/'result'), workers=1)
    infer(a)
    assert len(calls) == 5
    journal = tmp_path/'result/predictions.jsonl'
    with journal.open('a') as f: f.write('{"sample_id":')
    infer(a)
    assert len(calls) == 5  # Torn final journal write is repaired without repeating valid calls.
    assert json.loads((tmp_path/'result/status.json').read_text())['phase'] == 'inference_complete'
