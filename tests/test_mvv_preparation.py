import csv
import json

import pytest

from scripts.prepare_mvvbench import audit, source_path


def test_only_required_views_count_and_no_silent_question_drop(tmp_path):
    annotations = tmp_path/'questions.json'
    annotations.write_text(json.dumps({'s': [{'views': [0, 2]}, {'views': [0, 1]}]}))
    mapping = tmp_path/'mapping.csv'
    fields = ['target', 'sample_name', 'view', 'dataset', 'split', 'source_rel_path', 'source_file']
    with mapping.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for v in range(4):
            writer.writerow(dict(target=f's_view{v}.mp4', sample_name='s', view=v,
                dataset='panoptic', split='', source_rel_path=f's/v{v}.mp4', source_file=f'v{v}.mp4'))
    videos = tmp_path/'videos'; videos.mkdir()
    (videos/'s_view0.mp4').write_bytes(b'video')
    (videos/'s_view2.mp4').write_bytes(b'video')
    report = audit(annotations, mapping, videos, tmp_path/'out')
    assert report['required_videos'] == 3  # Mapping's unused camera 3 is irrelevant.
    assert report['total_questions'] == 2 and report['complete_questions'] == 1
    assert report['missing_videos'] == 1
    assert len(json.loads((tmp_path/'out/selected_questions.json').read_text())['s']) == 2
    annotations.write_text(json.dumps({'s': [{'views': [0, 99]}]}))
    report = audit(annotations, mapping, videos, tmp_path/'broken')
    assert report['unmapped_views'] == ['s_view99.mp4']
    assert report['complete_questions'] == 0 and report['total_questions'] == 1


def test_source_path_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        source_path(dict(dataset='egoexo', source_rel_path='../secret'), tmp_path)
