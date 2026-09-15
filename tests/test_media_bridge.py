from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest

from scripts.media_inputs import read_frames
from scripts.encode_media_geometry import scene_from_stages


def test_geometry_text_precision_is_explicit_and_preserves_saved_values():
    import json
    from core.prompts import make_messages
    scene = {'units':'m','coordinate_frame':'camera','objects':[
        {'instance_id':'x','category':'cup','center':[1.123456,2.,3.], 'size':[1.,1.,1.]}]}
    default = make_messages('q',scene)
    rounded = make_messages('q',scene,decimals=4)
    assert '1.123456' in default[1]['content']
    assert '1.1235' in rounded[1]['content'] and '1.123456' not in rounded[1]['content']
    assert scene['objects'][0]['center'][0] == 1.123456
    with pytest.raises(ValueError,match='decimals'):
        make_messages('q',scene,decimals=-1)


class Array(np.ndarray):
    def numpy(self):
        return np.asarray(self)


def test_video_sampling_retains_absolute_time(tmp_path):
    import cv2
    path = tmp_path/'clip.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 5, (32,32))
    for i in range(25):
        writer.write(np.full((32,32,3), i*5, dtype=np.uint8))
    writer.release()
    images, frames = read_frames({'media': {'video': {'path': str(path)}},
        'input_metadata': {'time_start': 1, 'time_end': 3}}, video_frames=4)
    assert len(images) == 4
    assert frames[0]['timestamp'] == 1 and frames[-1]['timestamp'] == 3
    assert all(a['timestamp'] < b['timestamp'] for a,b in zip(frames, frames[1:]))
    assert [f['frame_index'] for f in frames] == [5,8,12,15]
    with pytest.raises(ValueError, match='outside'):
        read_frames({'media': {'video': {'path': str(path)}}, 'input_metadata': {'time_start': 10}})


def test_multiview_order_does_not_invent_timestamps(tmp_path):
    paths = [tmp_path/'b.png',tmp_path/'a.png']
    for p in paths:
        Image.new('RGB',(8,8)).save(p)
    _,frames = read_frames({'media': {'image': [{'path':str(p)} for p in paths]}})
    assert [f['path'] for f in frames] == list(map(str,paths))
    assert all('timestamp' not in f for f in frames)


def test_metric_boxes_and_camera_motion_are_not_confused():
    pose0, pose1 = np.eye(4), np.eye(4)
    pose1[0,3] = 1
    stages = [{'pose':pose0, 'box':[0,0,2, .4,.8,1.2]},
              {'pose':pose1, 'box':[-.4,0,2, .4,.8,1.2]}]
    def boxes(stage):
        return (np.array([stage['box']]).view(Array), np.array([[1.,0,0,0]]).view(Array),
                np.array([.9]), np.array([True]))
    native = SimpleNamespace(_take_best_candidates=boxes, _take_predicted_pose_encoding=lambda s:s['pose'],
                             _pose_encoding_to_camera_pose=lambda pose,scale:pose)
    payload = {'open_labels':['cup'],'open_slot_ids':['cup_1']}
    frames = [{'view_id':'f0','timestamp':1.0},{'view_id':'f1','timestamp':2.0}]
    scene, _ = scene_from_stages(stages,payload,frames,native,{},metric_scale=2.5)
    obs = scene['tracks'][0]['observations']
    np.testing.assert_allclose(obs[0]['center'],obs[1]['center'])
    np.testing.assert_allclose(obs[0]['center'],[0,0,5])
    np.testing.assert_allclose(obs[0]['size'],[1,2,3])
    np.testing.assert_allclose(obs[0]['quaternion_xyzw'],[0,0,0,1])
    with pytest.raises(ValueError, match='output frames'):
        scene_from_stages(stages[:1],payload,frames,native,{})
