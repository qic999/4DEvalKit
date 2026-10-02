"""Regression checks for the perception adapter's external format boundary."""
import numpy as np
from PIL import Image
import pytest

from scripts.encode_wilddet3d_perception import read_image
from scripts.wilddet3d_protocol import native_to_official_box
from scripts.score_wilddet3d_perception import finite_metrics
from scripts.prepare_wilddet3d_perception import exclude_images


def test_anisotropic_rotated_box_preserves_physical_extents():
    # Explicit non-cube avoids the x/z swap being hidden by symmetry.
    from scipy.spatial.transform import Rotation
    center = np.array([1.4, -.3, 4.5])
    size = np.array([.8, 1.7, .4])
    q = Rotation.from_euler('xyz', [20, -30, 15], degrees=True).as_quat()[[3,0,1,2]]
    box = native_to_official_box(center, size, q)
    # Vis4D's documented OPENCV axes use length along x, height y, width z.
    w,l,h = box[3:6]
    signs = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    rotation = Rotation.from_quat(np.array(box[6:])[[1,2,3,0]]).as_matrix()
    actual = signs*np.array([l,h,w])/2 @ rotation.T + box[:3]
    expected = signs*size/2 @ Rotation.from_quat(q[[1,2,3,0]]).as_matrix().T + center
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_official_scannet_path_prefix_reads_original_rgb(tmp_path):
    root = tmp_path/'scannet'
    path = root/'val/scene/image/0.png'
    path.parent.mkdir(parents=True)
    Image.new('RGB', (23, 17), (17, 42, 9)).save(path)
    result = read_image(root, 'data/scannet/val/scene/image/0.png')
    assert result.size == (23,17)
    assert result.getpixel((0,0)) == (17,42,9)


def test_missing_image_raises_instead_of_silent_subset(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path, 'absent.jpg')


def test_undefined_official_metrics_are_not_reported_as_zero():
    assert finite_metrics({'AP': .25, 'APr': np.nan, 'ATE': np.inf}) == {'AP': .25, 'APr': None, 'ATE': None}


def test_explicit_subset_excludes_missing_gt_but_keeps_empty_images():
    from copy import deepcopy
    original = {'images': [{'id': 1, 'file_path': 'present.jpg'},
                           {'id': 2, 'file_path': 'missing.jpg'},
                           {'id': 3, 'file_path': 'empty.jpg'}],
                'annotations': [{'id': 10, 'image_id': 1}, {'id': 20, 'image_id': 2},
                                {'id': 21, 'image_id': 2}],
                'categories': [{'id': 7, 'name': 'chair'}], 'info': {'name': 'original'}}
    before = deepcopy(original)
    subset, scope = exclude_images(original, ['missing.jpg'])
    assert original == before
    assert [im['id'] for im in subset['images']] == [1, 3]
    assert subset['annotations'] == [{'id': 10, 'image_id': 1}]
    assert subset['categories'] == original['categories']
    assert scope['excluded_image_ids'] == [2]
    assert scope['evaluated_images'] == 2
    assert scope['excluded_annotations'] == 2
    with pytest.raises(ValueError, match='Unknown excluded'):
        exclude_images(original, ['typo.jpg'])
    with pytest.raises(ValueError, match='no evaluation images'):
        exclude_images(original, [im['file_path'] for im in original['images']])


def test_comparison_keeps_old_and_new_small_separate_across_runners(tmp_path):
    import csv
    from core.io import write_json
    from scripts.summarize_wilddet3d_perception import summarize
    write_json(tmp_path/'comparison_config.json', {
        'models': [{'name': 'full_e64', 'label': 'Full e64'},
                   {'name': 'small_e34', 'label': 'Old Small e34'},
                   {'name': 'small_e100', 'label': 'Small WDS518 e100'}],
        'status_files': ['status.json', 'old_small_status.json']})
    write_json(tmp_path/'status.json', {'jobs': []})
    write_json(tmp_path/'old_small_status.json', {
        'jobs': [{'dataset': 'stereo4d', 'model': 'small_e34', 'state': 'complete'}],
        'scoring': ['stereo4d/small_e34']})
    for model, score in [('full_e64', .10), ('small_e100', .25)]:
        write_json(tmp_path/'stereo4d'/model/'metrics.json', {'images': 383, 'metrics': {'dist': {'AP': score}}})
    summarize(tmp_path)
    text = (tmp_path/'comparison.md').read_text()
    assert '| Full e64 | Old Small e34 | Small WDS518 e100 |' in text
    assert '| stereo4d | AP (distance) | 10.00 | — | 25.00 |' in text
    rows = list(csv.DictReader((tmp_path/'comparison.csv').open()))
    old = next(row for row in rows if row['dataset'] == 'stereo4d' and row['model'] == 'small_e34')
    assert old['score_100'] == ''
    assert old['state'] == 'scoring'
