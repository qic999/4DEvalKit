import io
import json
from pathlib import Path

from PIL import Image
import pytest

from core.captions import CaptionStore, VERSION, caption_messages, media_identity
from core.inference import APIInferenceEngine
from core.io import digest, write_json
from core.observations import ObservationManifest
from scripts.run_caption_ablations import generate_caption, prepare_captions


def sample(tmp_path, sample_id='x'):
    image = tmp_path/'image.png'
    Image.new('RGB', (60, 80), 'blue').save(image)
    return {'sample_id': sample_id, 'question': 'SECRET QUESTION A. cup B. table',
            'answer': 'SECRET ANSWER', 'media': {'image': {'path': str(image)}},
            'input_metadata': {'choices': ['SECRET CHOICE'], 'target': 'SECRET TARGET'}}


def test_caption_generation_cannot_see_question_labels_or_boxes(tmp_path):
    row = sample(tmp_path); row['geometry'] = {'center': ['SECRET GEOMETRY']}
    messages = caption_messages(row)
    text = json.dumps(messages)
    assert 'SECRET' not in text and 'image_url' in text
    changed = dict(row, question='different', answer='different')
    assert caption_messages(changed) == messages
    assert media_identity(changed) == media_identity(row)
    duplicate = tmp_path/'copied.png'; duplicate.write_bytes((tmp_path/'image.png').read_bytes())
    changed['media'] = {'image': {'path': str(duplicate)}}
    assert media_identity(changed) == media_identity(row)


def bundle(tmp_path, manifest):
    path = tmp_path/'captions.json'
    data = {'schema_version': VERSION, 'config': {'model': 'test'},
            'media_manifest_digest': digest(manifest),
            'samples': [{'sample_id': r['sample_id'], 'caption': 'A blue cup on a table.',
                         'caption_key': 'same', 'status': 'ok', 'finish_reason': 'stop'}
                        for r in manifest['samples']]}
    write_json(path, data)
    return path, data


def test_caption_qa_uses_shared_text_without_loading_rgb(tmp_path, monkeypatch):
    row = sample(tmp_path); manifest = {'samples': [row]}; path = tmp_path/'manifest.json'
    write_json(path, manifest); captions, _ = bundle(tmp_path, manifest)
    import core.observations
    def forbidden(*args, **kwargs):
        raise AssertionError('Caption QA must not decode or send RGB')
    monkeypatch.setattr(core.observations, 'visual_content', forbidden)
    scene = {'objects': [{'category': 'geometry_only_object'}]}
    modes = {mode: ObservationManifest(path, mode=mode, captions=captions)
             for mode in ['caption', 'caption_boxes']}
    messages = {mode: obj.messages('x', row['question'], scene, max_chars=10000, decimals=4)
                for mode, obj in modes.items()}
    assert messages['caption'][0] == messages['caption_boxes'][0]
    assert messages['caption'][1]['content'][0] == messages['caption_boxes'][1]['content'][0]
    assert 'geometry_only_object' not in json.dumps(messages['caption'])
    assert 'geometry_only_object' in json.dumps(messages['caption_boxes'])
    assert all('image_url' not in json.dumps(m) for m in messages.values())
    assert modes['caption'].identity['captions'] == modes['caption_boxes'].identity['captions']
    with pytest.raises(ValueError, match='character limit'):
        modes['caption'].messages('x', row['question'], {}, max_chars=180, decimals=4)


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'length', 'manifest'])
def test_caption_integrity_fails_closed(tmp_path, failure):
    manifest = {'samples': [sample(tmp_path)]}; path, data = bundle(tmp_path, manifest)
    if failure == 'missing': data['samples'] = []
    if failure == 'duplicate': data['samples'] *= 2
    if failure == 'length': data['samples'][0]['finish_reason'] = 'length'
    if failure == 'manifest': data['media_manifest_digest'] = 'different'
    write_json(path, data)
    with pytest.raises(ValueError): CaptionStore(path, manifest)


def test_truncated_caption_retries_and_is_never_accepted(tmp_path):
    row = sample(tmp_path)
    class Engine:
        def __init__(self, finish): self.finish = finish
        def infer(self, messages):
            return {'status': 'ok' if self.finish == 'stop' else 'invalid_completion',
                    'raw_output': 'A blue object.', 'finish_reason': self.finish}
    cfg = {'max_image_pixels': 262144, 'max_tokens_attempts': [10, 20]}
    record = generate_caption(row, {}, 'key', cfg, [Engine('length'), Engine('stop')])
    assert record['status'] == 'ok' and len(record['attempts']) == 2
    record = generate_caption(row, {}, 'key', cfg, [Engine('length'), Engine('length')])
    assert record['status'] == 'failed' and not record['caption']


def test_duplicate_media_has_one_caption_and_resumes_without_inference(tmp_path, monkeypatch):
    import scripts.run_caption_ablations as module
    first = sample(tmp_path); second = dict(first, sample_id='y', question='Other question')
    manifest = tmp_path/'manifest.json'; write_json(manifest, {'samples': [first, second]})
    calls = []
    class Engine:
        def __init__(self, **kwargs): pass
        def infer(self, messages):
            calls.append(messages)
            return {'status': 'ok', 'finish_reason': 'stop', 'raw_output': 'A blue surface.'}
    monkeypatch.setattr(module, 'APIInferenceEngine', Engine)
    config = {'llm_name': 'test', 'llm_model': '/model', 'caption_concurrency': 2}
    job = {'name': 'test', 'manifest': str(manifest)}
    path = prepare_captions(job, config, tmp_path/'out', ['http://localhost:1/v1'], lambda **kw: None)
    initial = path.read_bytes(); data = json.loads(initial)
    assert len(calls) == 1 and data['unique_media'] == 1 and len(data['samples']) == 2
    assert data['samples'][0]['caption'] == data['samples'][1]['caption']
    prepare_captions(job, config, tmp_path/'out', ['http://localhost:1/v1'], lambda **kw: None)
    assert len(calls) == 1 and path.read_bytes() == initial


def test_replica_pool_distributes_identical_requests(monkeypatch):
    from core import inference
    sent = []
    def respond(request, **kwargs):
        sent.append((request.full_url, json.loads(request.data)))
        return io.StringIO(json.dumps({'choices': [{'message': {'content': 'A'}, 'finish_reason': 'stop'}]}))
    monkeypatch.setattr(inference.urllib.request, 'urlopen', respond)
    urls = ['http://localhost:1/v1', 'http://localhost:2/v1']
    engine = APIInferenceEngine(model='test', base_url=urls[0], base_urls=urls)
    messages = [{'role': 'user', 'content': 'Question'}]
    for _ in range(4): assert engine.infer(messages, structured_outputs={'choice': ['A', 'B']})['status'] == 'ok'
    assert [r[0] for r in sent] == [u+'/chat/completions' for u in urls*2]
    assert all(r[1] == sent[0][1] for r in sent)
