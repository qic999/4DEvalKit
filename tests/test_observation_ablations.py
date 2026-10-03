import base64
import io
import json

from PIL import Image
import pytest

from core.io import write_json
from core.observations import ObservationManifest
from core.response_constraints import response_constraint, matches_constraint
from core.inference import APIInferenceEngine


@pytest.mark.parametrize('question', [
    'q\n(A) left\n(B) right', 'q A. left B. right',
    'q\nOptions: A: left, B: right', 'q\nA. left\nB. right'])
def test_choice_constraints_use_public_option_letters(question):
    assert response_constraint('BLINK',question) == {'choice':['A','B']}
    assert not matches_constraint('The answer is A', {'choice':['A','B']})


def test_numeric_and_text_constraints_preserve_native_formats():
    constraint = response_constraint('Q-Spatial-Bench','distance?')
    assert matches_constraint(r'\scalar{0.42} \distance_unit{m}',constraint)
    assert not matches_constraint(r'\scalar{1e3} \distance_unit{m}',constraint)
    assert not matches_constraint('The distance is unknown',constraint)
    question = 'Choose between the following options: chair left or chair right'
    assert response_constraint('SAT',question,['chair right','chair left']) == {'choice':['chair left','chair right']}
    with pytest.raises(ValueError,match='public question'):
        response_constraint('SAT',question,['a hidden scoring label'])


def test_server_constraint_violation_is_not_accepted(monkeypatch):
    from core import inference
    def reply(req,**kwargs):
        body=json.loads(req.data)
        assert body['structured_outputs'] == {'choice':['A','B']}
        return io.StringIO(json.dumps({'choices':[{'message':{'content':'Explanation, then A'},'finish_reason':'stop'}]}))
    monkeypatch.setattr(inference.urllib.request,'urlopen',reply)
    engine=APIInferenceEngine(model='local',base_url='http://localhost:1/v1')
    result=engine.infer([{'role':'user','content':'q'}],structured_outputs={'choice':['A','B']})
    assert result['status']=='invalid_completion'


def test_rgb_box_arms_share_system_images_and_question_without_geometry_leakage(tmp_path):
    pic=tmp_path/'frame.jpg';Image.new('RGB',(1000,800),'red').save(pic)
    path=tmp_path/'manifest.json';question='Where?\nA. left\nB. right'
    write_json(path,{'samples':[{'sample_id':'x','question':question,'media':{'image':{'path':str(pic)}}}]})
    scene={'objects':[{'category':'cup','center':[1.23456,0,1], 'size':[1,1,1]}],
           'provenance':{'not_for_prompt':True},'units':'m','coordinate_frame':'camera'}
    messages={}
    for mode in ['boxes','rgb','rgb_boxes']:
        loader=ObservationManifest(path,mode=mode,max_pixels=65536)
        messages[mode]=loader.messages('x',question,scene,max_chars=10000,decimals=4)
    assert len({m[0]['content'] for m in messages.values()})==1
    assert 'cup' not in json.dumps(messages['rgb'])
    assert 'cup' in json.dumps(messages['boxes']) and 'cup' in json.dumps(messages['rgb_boxes'])
    assert 'image_url' not in json.dumps(messages['boxes'])
    image=lambda m:next(x['image_url']['url'] for x in m[1]['content'] if x['type']=='image_url')
    assert image(messages['rgb'])==image(messages['rgb_boxes'])
    decoded=Image.open(io.BytesIO(base64.b64decode(image(messages['rgb']).split(',')[1])))
    assert decoded.width*decoded.height <= 65536
    assert all(question in json.dumps(m).replace('\\n','\n') for m in messages.values())
    assert all('not_for_prompt' not in json.dumps(m) for m in messages.values())
    assert scene['objects'][0]['center'][0] == 1.23456
    with pytest.raises(ValueError,match='question changed'):
        loader.messages('x','modified question',scene,max_chars=10000,decimals=4)


def test_ablation_gate_requires_complete_nontruncated_native_answers(tmp_path):
    from scripts.run_observation_ablations import audit_answers
    from core.response_constraints import VERSION
    manifest=tmp_path/'manifest.json'
    write_json(manifest,{'samples':[{'sample_id':'a'},{'sample_id':'b'}]})
    write_json(tmp_path/'config.json',{'jobs':[{'name':'job','manifest':str(manifest)}],
                                     'variants':[{'name':'full'}]})
    rows=[{'sample_id':key,'status':'ok','finish_reason':'stop','raw_output':'A',
           'response_constraint':{'choice':['A','B']}} for key in ['a','b']]
    result={'results':rows,'config':{'answer_format':VERSION},'primary_metric':{'score_100':0}}
    path=tmp_path/'job/full/qa.json';write_json(path,result)
    assert audit_answers(tmp_path)['total_responses']==2
    rows[0]['finish_reason']='length';write_json(path,result)
    with pytest.raises(ValueError,match='invalid/truncated'):
        audit_answers(tmp_path)
    result['results']=rows[:1];write_json(path,result)
    with pytest.raises(ValueError,match='Incomplete'):
        audit_answers(tmp_path)


@pytest.mark.parametrize('mode', ['rgb', 'rgb_boxes'])
def test_shared_visual_cache_preserves_questions_and_invalidates_inputs(tmp_path, monkeypatch, mode):
    from core import observations
    original = observations.visual_content
    calls = []
    def counted(row, **kwargs):
        calls.append(row['sample_id'])
        return original(row, **kwargs)
    monkeypatch.setattr(observations, 'visual_content', counted)
    pic = tmp_path/'frame.png'; Image.new('RGB', (20, 20), 'red').save(pic)
    rows = [dict(sample_id=key, question=f'Question {key}?',
                 media={'image': {'path': str(pic)}}) for key in ['a', 'b', 'c']]
    rows[2]['input_metadata'] = {'frame_indices': [0]}
    path = tmp_path/'manifest.json'; write_json(path, {'samples': rows})
    loader = ObservationManifest(path, mode=mode)
    def messages(i):
        row = rows[i]
        return loader.messages(row['sample_id'], row['question'], {}, max_chars=10000, decimals=4)
    first, second = messages(0), messages(1)
    assert calls == ['a']
    assert first[1]['content'][:-1] == second[1]['content'][:-1] == original(rows[0])
    assert 'Question a?' in first[1]['content'][-1]['text']
    assert 'Question b?' in second[1]['content'][-1]['text']
    assert 'Question a?' not in second[1]['content'][-1]['text']
    first[1]['content'][1]['image_url']['url'] = 'mutated by caller'
    assert messages(1) == second  # Returned messages cannot corrupt the cache.
    messages(2)
    assert calls == ['a', 'c']  # Sampling metadata is part of observation identity.
    Image.new('RGB', (31, 20), 'blue').save(pic)
    changed = messages(2)
    assert calls == ['a', 'c', 'c']
    assert changed[1]['content'][:-1] == original(rows[2])
    assert changed[1]['content'][:-1] != second[1]['content'][:-1]
