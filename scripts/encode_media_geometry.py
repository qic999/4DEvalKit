"""Run Full/Small on exported RGB media and predicted first-frame proposals.

This protocol uses no benchmark 2D/3D labels, depth, or camera poses. Fixed
vocabulary proposals initialize native object slots. Video observations retain
source timestamps and predicted camera motion; no cross-clip tracks are assumed.
"""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time

import numpy as np
from PIL import Image

from core.geometry import normalize_scene
from core.io import digest, read_json, source_signature, write_json
from core.runner import output_lock
from scripts.media_inputs import read_frames


def make_datapoint(images, frames, record, native):
    import torch
    from sam3.train.data.sam3_image_dataset import Datapoint, Image as ModelImage
    from inference_gt2d.open_vocab_slots import apply_open_vocab_slots
    from depth_anything_3.utils.io.input_processor import InputProcessor
    resized = [im.resize((1024, 1024), Image.Resampling.LANCZOS if max(im.size)>1024 else Image.Resampling.BICUBIC)
               for im in images]
    spatial, _, _ = InputProcessor()([np.asarray(im) for im in resized], None, None, 504, 'upper_bound_resize')
    mean = torch.tensor([.485,.456,.406]).view(3,1,1)
    std = torch.tensor([.229,.224,.225]).view(3,1,1)
    model_images = []
    for i,im in enumerate(resized):
        tensor = torch.from_numpy(np.array(im)).permute(2,0,1).float()/255
        model_images.append(ModelImage(data=(tensor-mean)/std, objects=[], size=(1024,1024),
                                       spatial_data=spatial[i].float().contiguous()))
    count = len(images)
    # These identity matrices are loss-side placeholders required by native
    # object dataclasses, never used/exported as observed or predicted cameras.
    payload = {'scene_id': record['sample_id'], 'frame_paths': [f['view_id'] for f in frames],
               'intrinsics': torch.eye(3).repeat(count,1,1), 'extrinsics': torch.eye(4).repeat(count,1,1),
               'T_gravities': torch.eye(4).repeat(count,1,1)}
    data = Datapoint(find_queries=[], images=model_images, raw_images=resized, reference_payload=payload)
    class Provider:
        def load(self, scene_id, frame_path, clip_idx, image_h, image_w):
            scale = np.array([image_w/record['image_size'][0], image_h/record['image_size'][1]]*2)
            return [{**d, 'bbox_xyxy': np.asarray(d['bbox_xyxy'])*scale, 'source_label': d['label']}
                    for d in record['detections']]
    created, _ = apply_open_vocab_slots(data, Provider(), 0, max_objects=1000, min_score=0, min_box_area=1)
    if created == 0:
        return None, payload
    # Native tracking counts stages from find_queries, not img_batch. Register
    # every observed frame. Only frame zero has conditioning boxes; later
    # loss-side boxes are empty and the tracker propagates its own predictions.
    from dataclasses import replace
    initial_queries = data.find_queries[:]
    for frame_index in range(1, count):
        data.images[frame_index].objects = [replace(obj, frame_index=frame_index,
            bbox=torch.zeros(4), area=0.0) for obj in data.images[0].objects]
        for query in initial_queries:
            data.find_queries.append(replace(query, image_id=frame_index,
                query_processing_order=frame_index, input_bbox=None, input_bbox_label=None,
                inference_metadata=replace(query.inference_metadata, frame_index=frame_index)))
    return native.collate_one(data), payload


def scene_from_stages(stages, payload, frames, native, provenance, metric_scale=2.5):
    from scipy.spatial.transform import Rotation
    dynamic = 'timestamp' in frames[0]
    scene = {'units': 'm', 'coordinate_frame': 'predicted_first_camera_x_right_y_down_z_forward' if dynamic
             else 'view_local_camera_x_right_y_down_z_forward', 'objects': [], 'provenance': provenance}
    if len(stages) != len(frames):
        raise ValueError(f'Expected {len(frames)} output frames, got {len(stages)}')
    poses = []
    for stage in stages:
        encoding = native._take_predicted_pose_encoding(stage)
        if encoding is None:
            if dynamic or len(frames) > 1:
                raise ValueError('Multi-frame input has no predicted camera poses')
            poses.append(np.eye(4))
        else:
            poses.append(native._pose_encoding_to_camera_pose(encoding, metric_scale))
    gauge = np.linalg.inv(poses[0])
    poses = [gauge @ pose for pose in poses]
    tracks, views, cameras, invalid = {}, [], [], 0
    raw = []
    for stage,frame,pose in zip(stages,frames,poses):
        boxes, quats, scores, present = native._take_best_candidates(stage)
        if len(boxes) != len(payload['open_labels']):
            raise ValueError('Native slot count does not match predicted proposals')
        objects = []
        for i,(box,quat,score,visible) in enumerate(zip(boxes,quats,scores,present)):
            if not bool(visible):
                continue
            box, quat = box.numpy(), quat.numpy()
            if not np.isfinite(box[:6]).all() or not np.isfinite(quat).all() or min(box[3:6])<=0 or np.linalg.norm(quat)<1e-8:
                invalid += 1
                continue
            center, size = box[:3]*metric_scale, box[3:6]*metric_scale
            rotation = Rotation.from_quat(quat[[1,2,3,0]])
            slot, label = payload['open_slot_ids'][i], payload['open_labels'][i]
            obj = {'instance_id': slot, 'category': label, 'center': center.tolist(), 'size': size.tolist(),
                   'quaternion_xyzw': rotation.as_quat().tolist(), 'confidence': float(score), 'view_id': frame['view_id']}
            raw.append({'frame': frame, 'slot_id': slot, 'bbox_normalized': box[:6].tolist(),
                        'quaternion_wxyz': quat.tolist(), 'score': float(score)})
            if dynamic:
                observation = {'timestamp': frame['timestamp'], 'view_id': frame['view_id'],
                    'center': (pose[:3,:3] @ center + pose[:3,3]).tolist(), 'size': size.tolist(),
                    'quaternion_xyzw': (Rotation.from_matrix(pose[:3,:3])*rotation).as_quat().tolist(),
                    'confidence': float(score)}
                tracks.setdefault(slot, {'track_id': slot, 'category': label, 'observations': []})['observations'].append(observation)
            else:
                objects.append(obj)
        camera = {'view_id': frame['view_id'], 'camera_to_world': pose.tolist()}
        if dynamic:
            camera['timestamp'] = frame['timestamp']
        cameras.append(camera)
        if not dynamic:
            views.append({'view_id': frame['view_id'], 'objects': objects, 'camera_to_world': pose.tolist()})
    scene['cameras'] = cameras
    if dynamic:
        scene['tracks'] = list(tracks.values())
    elif len(views)==1:
        scene['objects'] = views[0]['objects']
    else:
        scene['views'] = views
    return normalize_scene(scene), {'invalid_box_observations': invalid, 'raw_observations': raw}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--proposals', required=True)
    p.add_argument('--model-repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--profile', choices=['full','lite'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--wait-inputs', action='store_true', help='Consume proposal records as the detector writes them')
    args = p.parse_args()
    sys.path.insert(0, str(Path(args.model_repo).resolve()))
    import torch
    from inference_gt2d import scene_inference as native
    from sam3.model.utils.misc import copy_data_to_device
    from scripts.encoder_entry import install_box_only_detector_fusion, install_tracker_mask_offload, install_chunked_rope, install_large_interpolation, install_depth_scale_sampling_fix
    index = read_json(Path(args.proposals)/'index.json')
    rows = {r['sample_id']: r for r in read_json(args.manifest)['samples']}
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = index['records'][args.shard_index::args.num_shards]
    provenance = {'encoder': 'spatial_encoder_v2' if args.profile=='full' else 'spatial_encoder_v2_small',
        'checkpoint': source_signature(args.checkpoint), 'geometry_source': 'predicted',
        'proposal_source': 'groundingdino_fixed_vocabulary', 'label_source': 'detector',
        'camera_pose_source': 'predicted', 'protocol': 'fixed_vocab_first_frame_v1',
        'proposal_config': {k:v for k,v in index['config'].items() if k!='sample_ids'},
        'proposal_config_sha256': digest(index['config']), 'normalized_box_to_meters': 2.5,
        'limitations': 'First sampled frame initializes slots; later entrants may be missed. No cross-clip identity stitching.'}
    config = {'manifest': str(Path(args.manifest).resolve()), 'provenance': provenance,
              'shard_index': args.shard_index, 'num_shards': args.num_shards}
    with output_lock(out/'geometry.json'):
        if (out/'config.json').exists() and read_json(out/'config.json') != config:
            raise ValueError('Encoder configuration changed; use a fresh output')
        write_json(out/'config.json', config)
        model_args = SimpleNamespace(sam3_checkpoint=None, checkpoint=args.checkpoint, device='cuda',
            model_profile=args.profile, spatial_resolution=504, multiplex_count=1, use_fa3=False,
            use_act_checkpoint_multiplex_transformer=True, bbox_head_mode='reference_per_candidate',
            max_cond_frames_in_attn=-1, use_maskmem_tpos_v2=False, use_linear_no_obj_ptr=False)
        if args.profile == 'full':
            install_depth_scale_sampling_fix()
        model = native.make_model(model_args)
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        weights = checkpoint.get('model', checkpoint)
        actual = model.state_dict()
        if weights.keys() != actual.keys() or any(weights[k].shape != actual[k].shape for k in actual):
            raise ValueError('Checkpoint does not exactly match the encoder architecture')
        del checkpoint, weights, actual
        install_large_interpolation()
        install_chunked_rope(inplace_k=True)
        if args.profile == 'full':
            install_box_only_detector_fusion(model)
        install_tracker_mask_offload(model)
        scenes = {}
        for path in records:
            if args.wait_inputs:
                from scripts.finish_scannet_recovery import process_identity
                deadline = time.monotonic() + 24*3600
                while not Path(path).is_file():
                    producer = read_json(Path(args.proposals)/'status.json')
                    if process_identity(producer['pid']) is None or time.monotonic()>deadline:
                        raise RuntimeError(f'Proposal producer stopped before writing {path}')
                    time.sleep(2)
            record = read_json(path)
            sample_id = record['sample_id']
            result_path = out/'samples'/(sample_id.replace(':','_')+'.json')
            fingerprint = digest({'proposal': record, 'config': config})
            if result_path.exists():
                result = read_json(result_path)
                if result['fingerprint'] != fingerprint:
                    raise ValueError('Input changed since previous geometry inference')
            else:
                images, frames = read_frames(rows[sample_id], index['config']['video_frames'])
                if frames != record['frames']:
                    raise ValueError('Media sampling changed since proposal generation')
                if not record['detections']:
                    result = {'scene': normalize_scene({'units':'m','coordinate_frame':'camera_x_right_y_down_z_forward',
                              'objects': [], 'provenance': provenance}), 'diagnostics': {'no_proposals': True}}
                else:
                    batch, payload = make_datapoint(images, frames, record, native)
                    if batch is None:
                        raise ValueError('All predicted proposals were rejected by native slot creation')
                    batch = copy_data_to_device(batch, torch.device('cuda'), non_blocking=True)
                    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        output = model(batch, is_inference=True)
                    stages = native.unwrap_model_output(output)
                    scene, diagnostics = scene_from_stages(stages, payload, frames, native, provenance)
                    result = {'scene': scene, 'diagnostics': diagnostics}
                    del batch, output, stages
                    torch.cuda.empty_cache()
                result.update(sample_id=sample_id, fingerprint=fingerprint)
                write_json(result_path, result)
            scenes[sample_id] = result['scene']
            write_json(out/'status.json', {'pid': os.getpid(), 'phase': 'encoding', 'completed': len(scenes),
                                          'total': len(records), 'updated': time.time()})
            print(f'{sample_id}: geometry saved ({len(scenes)}/{len(records)})', flush=True)
        write_json(out/'geometry.json', {'scenes': scenes})
        write_json(out/'status.json', {'pid': os.getpid(), 'phase': 'complete', 'completed': len(scenes), 'total': len(records)})


if __name__ == '__main__':
    main()
