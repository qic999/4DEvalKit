from copy import deepcopy
import pytest

from core.io import write_json
from core.response_constraints import VERSION
from scripts.run_checkpoint_evaluation import audit_result, merge_geometry


def test_merge_rejects_duplicate_missing_and_unexpected_samples(tmp_path):
    scene = {'units': 'm', 'coordinate_frame': 'camera', 'objects': []}
    a, b = tmp_path/'a.json', tmp_path/'b.json'
    write_json(a, {'scenes': {'a': scene}})
    write_json(b, {'scenes': {'b': scene}})
    assert set(merge_geometry([a, b], {'a', 'b'})['scenes']) == {'a', 'b'}
    with pytest.raises(ValueError, match='Incomplete'):
        merge_geometry([a], {'a', 'b'})
    with pytest.raises(ValueError, match='unexpected'):
        merge_geometry([a, b], {'a'})
    with pytest.raises(ValueError, match='Duplicate'):
        merge_geometry([a, a], {'a'})


def test_answer_audit_rejects_truncation_and_wrong_observations():
    from core.response_constraints import response_constraint
    constraint = response_constraint('CV-Bench', 'Choose an answer.\nA. one\nB. two')
    result = {'num_samples': 1, 'config': {'answer_format': VERSION,
              'observations': {'mode': 'caption_boxes', 'captions': {'digest': 'frozen'}}},
              'results': [{'sample_id': 'a', 'status': 'ok', 'finish_reason': 'stop',
                           'raw_output': 'A', 'response_constraint': constraint}]}
    audit_result(result, {'a'}, 'caption_boxes', 'frozen')
    for mutated in ('length', 'empty'):
        bad = deepcopy(result)
        if mutated == 'length': bad['results'][0]['finish_reason'] = 'length'
        else: bad['results'][0]['raw_output'] = ''
        with pytest.raises(ValueError, match='truncated'):
            audit_result(bad, {'a'}, 'caption_boxes', 'frozen')
    with pytest.raises(ValueError, match='caption'):
        audit_result(result, {'a'}, 'caption_boxes', 'different')
    with pytest.raises(ValueError, match='protocol'):
        audit_result(result, {'a'}, 'boxes')
    duplicate = deepcopy(result); duplicate['results'] *= 2
    with pytest.raises(ValueError, match='coverage'):
        audit_result(duplicate, {'a', 'b'}, 'caption_boxes', 'frozen')
