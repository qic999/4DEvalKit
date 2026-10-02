"""Single-image SpatialEncoder perception: RGB + GT 2D box, no GT 3D/depth/K."""
import argparse
import io
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.io import digest, read_json, source_signature, write_json
from core.runner import output_lock
from scripts.encode_media_geometry import make_datapoint
from scripts.wilddet3d_protocol import native_to_official_box


def read_image(root, relative):
    prefix = f'data/{Path(root).name}/'
    if relative.startswith(prefix):
        relative = relative[len(prefix):]
    path = Path(root)/relative
    if path.is_file():
        return Image.open(path).convert('RGB')
    # Vis4D HDF5Backend layout: val/a/b.jpg -> val.hdf5['a/b.jpg'].
    parts = Path(relative).parts
    for i in range(1, len(parts)):
        candidate = Path(root).joinpath(*parts[:i]).with_suffix('.hdf5')
        if candidate.is_file():
            import h5py
            with h5py.File(candidate, 'r') as h:
                value = h['/'.join(parts[i:])][()]
                return Image.open(io.BytesIO(bytes(value))).convert('RGB')
    raise FileNotFoundError(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['manifest', 'model-repo', 'checkpoint', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--profile', choices=['full', 'lite'], required=True)
    p.add_argument('--resolution', type=int, default=518)
    p.add_argument('--spatial-resolution', type=int, default=518)
    p.add_argument('--model-image-size', type=int, default=518)
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--prompt-batch-size', type=int, default=32)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards or args.prompt_batch_size < 1:
        p.error('Invalid shard or prompt batch size')
    manifest = read_json(args.manifest)
    config = {**vars(args), 'manifest_signature': source_signature(args.manifest),
              'checkpoint_signature': source_signature(args.checkpoint),
              'metric_scale': 2.5, 'inputs': 'RGB + GT 2D box prompts; GT category labels for slot bookkeeping',
              'no_gt_inputs': ['3D boxes', 'depth', 'intrinsics', 'camera poses'],
              'selection': 'one best predicted bbox IoU candidate per oracle prompt; no score/presence threshold; no GT 3D selection'}
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = manifest['samples'][args.shard_index::args.num_shards]
    if args.limit is not None:
        rows = rows[:args.limit]
    sys.path.insert(0, str(Path(args.model_repo).resolve()))
    import torch
    from inference_gt2d import scene_inference as native
    from sam3.model.utils.misc import copy_data_to_device
    from scripts.encoder_entry import (install_box_only_detector_fusion, install_tracker_mask_offload,
        install_chunked_rope, install_large_interpolation, install_depth_scale_sampling_fix)
    with output_lock(out/f'shard_{args.shard_index}.json'):
        cfgpath = out/f'config_{args.shard_index}.json'
        if cfgpath.exists() and read_json(cfgpath) != config:
            raise ValueError('Configuration changed; use a new output directory')
        write_json(cfgpath, config)
        if args.profile == 'full':
            install_depth_scale_sampling_fix()
        model_args = SimpleNamespace(sam3_checkpoint=None, checkpoint=args.checkpoint, device='cuda',
            model_profile=args.profile, spatial_resolution=args.spatial_resolution,
            model_image_size=args.model_image_size, multiplex_count=1, use_fa3=False,
            use_act_checkpoint_multiplex_transformer=True, bbox_head_mode='reference_per_candidate',
            max_cond_frames_in_attn=-1, use_maskmem_tpos_v2=False, use_linear_no_obj_ptr=False)
        model = native.make_model(model_args)
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
        weights = ckpt.get('model', ckpt)
        actual = model.state_dict()
        if weights.keys() != actual.keys() or any(weights[k].shape != actual[k].shape for k in actual):
            raise ValueError('Checkpoint does not exactly match model architecture')
        del weights, ckpt, actual
        install_large_interpolation()
        install_chunked_rope(inplace_k=True)
        if args.profile == 'full':
            install_box_only_detector_fusion(model)
        install_tracker_mask_offload(model)
        start = time.monotonic()
        for n, row in enumerate(rows, 1):
            path = out/'samples'/f'{row["image_id"]}.json'
            fingerprint = digest({'row': row, 'config': config})
            if path.exists():
                if read_json(path)['fingerprint'] != fingerprint:
                    raise ValueError('Prediction fingerprint changed')
                continue
            im = read_image(manifest['data_root'], row['file_path'])
            if list(im.size) != row['image_size']:
                raise ValueError(f'Original RGB size mismatch: {row["file_path"]}: {im.size} vs {row["image_size"]}')
            predictions, rejected = [], 0
            for offset in range(0, len(row['detections']), args.prompt_batch_size):
                dets = row['detections'][offset:offset+args.prompt_batch_size]
                # A fixed neutral query avoids category canonicalization/drop logic.
                # GT category IDs are attached only after forward for oracle-box scoring.
                record = {'sample_id': str(row['image_id']), 'image_size': row['image_size'],
                          'detections': [{'bbox_xyxy': d['bbox_xyxy'], 'score': 1.0, 'label': 'foreground'} for d in dets]}
                frames = [{'view_id': row['file_path']}]
                batch, payload = make_datapoint([im], frames, record, native,
                    resolution=args.resolution, spatial_resolution=args.spatial_resolution)
                if batch is None or len(payload['open_slot_ids']) != len(dets):
                    raise ValueError('Native slot filtering changed official oracle prompts')
                batch = copy_data_to_device(batch, torch.device('cuda'), non_blocking=True)
                with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    output = model(batch, is_inference=True)
                stages = native.unwrap_model_output(output)
                if len(stages) != 1:
                    raise ValueError('Expected one stage for a single image')
                boxes, quats, scores, present = native._take_best_candidates(stages[0])
                if len(boxes) != len(dets):
                    raise ValueError('Output slot count mismatch')
                for d, b, q, score, visible in zip(dets, boxes, quats, scores, present):
                    b, q = b.numpy(), q.numpy()
                    if not (np.isfinite(b[:6]).all() and np.isfinite(q).all() and np.isfinite(float(score))):
                        raise ValueError('Nonfinite model prediction')
                    if min(b[3:6]) <= 0 or np.linalg.norm(q) < 1e-8:
                        rejected += 1
                        continue
                    q = q/np.linalg.norm(q)
                    predictions.append({'category_id': d['category_id'], 'bbox_xyxy': d['bbox_xyxy'],
                        'score': float(score), 'box3d_wlh_wxyz': native_to_official_box(
                            (b[:3]*2.5).tolist(), (b[3:6]*2.5).tolist(), q.tolist())})
                del batch, output, stages
                torch.cuda.empty_cache()
            write_json(path, {'image_id': row['image_id'], 'fingerprint': fingerprint,
                             'predictions': predictions, 'rejected_predictions': rejected,
                             'prompt_count': len(row['detections'])})
            status = {'completed': n, 'total': len(rows), 'elapsed_seconds': time.monotonic()-start,
                      'last_image_id': row['image_id']}
            write_json(out/f'status_{args.shard_index}.json', status)
            print(status, flush=True)
        write_json(out/f'shard_{args.shard_index}.json', {'complete': True, 'image_ids': [r['image_id'] for r in rows]})


if __name__ == '__main__':
    main()
