import json

import pytest

from benchmark.loader import BenchmarkSession, media_manifest
from benchmark.temporal import TemporalDataset, tempcompass_match, fourd_option
from scripts.benchmark_registry import get_spec
from scripts.encode_media_geometry import combine_camera_scenes, make_datapoint


def test_motionbench_hidden_test_not_scored(tmp_path):
    p = tmp_path/'meta.jsonl'
    rows = [dict(question_type='motion', video_path='v.mp4', qa=[
        dict(uid='known', question='Where?\nA. left\nB. right', answer='A'),
        dict(uid='hidden', question='Where?\nA. left\nB. right', answer='NA')])]
    p.write_text('\n'.join(json.dumps(r) for r in rows))
    s = BenchmarkSession(get_spec('MotionBench'), data=str(p))
    assert s.selected_count == 1
    samples = next(s.batches())
    assert 'answer' not in json.dumps(media_manifest(samples[0]))
    adapter = s.adapter
    assert adapter.compute_statistics(adapter.evaluate_results([samples[0]['sample']], ['B']))['overall_accuracy'] == 0


def test_clevrer_whole_question_groups_and_scoring(tmp_path):
    p = tmp_path/'val.json'
    p.write_text(json.dumps([dict(scene_index=10, video_filename='v.mp4', questions=[dict(
        question_id=1, question='Why?', question_type='explanatory', choices=[
            dict(choice_id=0, choice='Collision', answer='correct'),
            dict(choice_id=1, choice='Magic', answer='wrong')])])]))
    s = BenchmarkSession(get_spec('CLEVRER'), data=str(p), limit=1)
    assert s.selected_count == 2
    samples = [x['sample'] for b in s.batches() for x in b]
    result = s.adapter.evaluate_results(samples, ['A', 'A'])
    stats = s.adapter.compute_statistics(result)
    assert stats['overall_accuracy'] == .5
    assert stats['overall_question_accuracy'] == 0
    with pytest.raises(ValueError, match='Incomplete'):
        s.adapter.compute_statistics(result[:1])


def test_upstream_matching_edge_cases():
    assert tempcompass_match('A) running', 'A. running')
    assert not tempcompass_match('(A)', 'A. running')
    assert not tempcompass_match('The answer is A', 'A. running')
    assert fourd_option('(B) but also (A)') == 'B'
    assert fourd_option('C.') == 'C'
    assert fourd_option('Cat') is None


def test_multi_camera_gauges_and_temporal_identity():
    def scene(view):
        return dict(cameras=[dict(view_id=view, camera_to_world=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])],
                    tracks=[dict(track_id='0', category='car', observations=[dict(view_id=view,
                        timestamp=1.0, center=[0,0,3], size=[1,1,1])])])
    merged = combine_camera_scenes([('1', scene('cam1')), ('8', scene('cam8'))], {})
    assert 'tracks' not in merged  # No fabricated shared world or cross-camera identity.
    ids = [v['objects'][0]['instance_id'] for v in merged['views']]
    assert ids == ['camera_1/0', 'camera_8/0']
    assert all(v['objects'][0]['timestamp'] == 1 for v in merged['views'])
    with pytest.raises(ValueError, match='separately'):
        make_datapoint([], [{'camera_id':'1'}, {'camera_id':'8'}], {}, None)


def test_mvv_grounding_labels_are_not_inputs(tmp_path):
    p = tmp_path/'qa.json'
    p.write_text(json.dumps({'s': [dict(views=[2,1], question='Which?', choices={'A':'x','B':'y'},
        correct_answer='B', gt_answer_text='y', category='count', static_or_dynamic='dynamic',
        timestamp_start=3, timestamp_end=4)]}))
    s = BenchmarkSession(get_spec('MVVBench'), data=str(p))
    m = media_manifest(next(s.batches())[0])
    assert m['input_metadata']['view_ids'] == ['2','1']
    assert 'time_start' not in m['input_metadata']
    assert 'correct_answer' not in json.dumps(m)


def test_frozen_subset_retains_original_denominator_metadata(tmp_path):
    p=tmp_path/'subset.json'
    p.write_text(json.dumps(dict(schema_version='4deval.temporal.v1',benchmark='MotionBench',
        source={'available':4018},samples=[dict(id='q',question='Which?',choices={'A':'x','B':'y'},
        answer='A',video='v.mp4',question_type='motion')])))
    s=BenchmarkSession(get_spec('MotionBench'),data=str(p))
    assert s.available_count==4018 and s.selected_count==1


def test_physion_fixed_features_preserve_repeated_initial_observation():
    import numpy as np
    from scripts.physion_box_features import box_features
    obs=[dict(view_id=f'frame_{i}',center=[i,0,2],size=[1,2,3]) for i in [0,15]]
    features=box_features({'tracks':[{'observations':obs}]},[0,0,0,15])
    assert features.shape==(4,45)
    assert np.array_equal(features[0],features[1]) and np.array_equal(features[1],features[2])
    assert features[3,0]==15 and features[0,-1]==1
    assert np.isfinite(box_features({},[0,15,30,45])).all()
