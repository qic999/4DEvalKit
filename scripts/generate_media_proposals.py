"""GroundingDINO proposals from public images only; no questions or answers.

Videos are initialized from their first sampled frame. Later entrants may be
missed; this limitation is recorded explicitly for the geometry protocol.
"""
import argparse
import inspect
import os
from pathlib import Path
import time

from core.io import digest, read_json, write_json
from core.runner import output_lock
from scripts.media_inputs import read_frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--vocabulary', default='configs/detector_vocabulary.json')
    p.add_argument('--limit', type=int)
    p.add_argument('--video-frames', type=int, default=16)
    p.add_argument('--max-objects', type=int, default=40)
    p.add_argument('--box-threshold', type=float, default=.25)
    p.add_argument('--text-threshold', type=float, default=.20)
    args = p.parse_args()
    import torch
    from torchvision.ops import nms
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    vocab = read_json(args.vocabulary)
    manifest = read_json(args.manifest)
    rows = manifest['samples'][:args.limit]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    identity = {'manifest': str(Path(args.manifest).resolve()), 'model': str(Path(args.model).resolve()),
                'vocabulary': vocab, 'video_frames': args.video_frames, 'max_objects': args.max_objects,
                'box_threshold': args.box_threshold, 'text_threshold': args.text_threshold,
                'sample_ids': [r['sample_id'] for r in rows], 'protocol': 'fixed_vocab_first_frame_v1'}
    with output_lock(out / 'index.json'):
        if (out / 'config.json').exists() and read_json(out / 'config.json') != identity:
            raise ValueError('Proposal configuration changed; choose a new output')
        write_json(out / 'config.json', identity)
        paths = [str(out/'samples'/(r['sample_id'].replace(':','_')+'.json')) for r in rows]
        write_json(out/'status.json', {'pid': os.getpid(), 'phase': 'loading_detector', 'completed': 0, 'total':len(rows)})
        write_json(out/'index.json', {'config':identity, 'records':paths, 'complete':False})
        processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model, local_files_only=True).eval().to('cuda')
        records = []
        last_fingerprint, last_record = None, None
        postprocess = processor.post_process_grounded_object_detection
        threshold_arg = 'box_threshold' if 'box_threshold' in inspect.signature(postprocess).parameters else 'threshold'
        for row in rows:
            path = out / 'samples' / (row['sample_id'].replace(':', '_') + '.json')
            fingerprint = digest({'media': row['media'], 'time': row.get('input_metadata', {}), 'config': identity})
            if path.exists():
                record = read_json(path)
                if record['fingerprint'] != fingerprint:
                    raise ValueError('Proposal sample changed')
            else:
                if fingerprint == last_fingerprint:
                    record = dict(last_record, sample_id=row['sample_id'])
                    write_json(path, record)
                    records.append(str(path))
                    continue
                images, frames = read_frames(row, args.video_frames)
                first_frames = [i for i, frame in enumerate(frames)
                                if i == 0 or frame.get('camera_id') != frames[i-1].get('camera_id')]
                views = []
                for frame_index in first_frames:
                    image = images[frame_index]
                    detections = detect(image, vocab, processor, model, postprocess, threshold_arg, args, torch, nms)
                    views.append({'camera_id': frames[frame_index].get('camera_id'),
                                  'image_size': [image.width, image.height], 'detections': detections})
                record = {'sample_id': row['sample_id'], 'fingerprint': fingerprint, 'frames': frames,
                          'image_size': views[0]['image_size'], 'detections': views[0]['detections']}
                if len(views) > 1:
                    record['camera_proposals'] = views
                write_json(path, record)
                del images
            last_fingerprint, last_record = fingerprint, record
            records.append(str(path))
            write_json(out / 'status.json', {'pid': os.getpid(), 'phase': 'proposals',
                'completed': len(records), 'total': len(rows), 'updated': time.time()})
            print(f"{row['sample_id']}: {len(record['detections'])} proposals", flush=True)
        write_json(out / 'index.json', {'config': identity, 'records': records, 'complete': True})
        write_json(out / 'status.json', {'pid': os.getpid(), 'phase': 'complete', 'completed': len(records), 'total': len(rows)})


def detect(image, vocab, processor, model, postprocess, threshold_arg, args, torch, nms):
    boxes, scores, labels = [], [], []
    for start in range(0, len(vocab), 30):
        text = ' '.join(label + '.' for label in vocab[start:start+30])
        inputs = processor(images=image, text=text, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            outputs = model(**inputs)
        result = postprocess(outputs, inputs.input_ids,
            target_sizes=[(image.height, image.width)], text_threshold=args.text_threshold,
            **{threshold_arg: args.box_threshold})[0]
        boxes.extend(result['boxes'].detach().float().cpu().tolist())
        scores.extend(result['scores'].detach().float().cpu().tolist())
        labels.extend(result['text_labels'])
    detections = []
    if boxes:
        keep = nms(torch.tensor(boxes), torch.tensor(scores), .65)[:args.max_objects]
        detections = [{'bbox_xyxy': boxes[i], 'score': scores[i], 'label': labels[i]} for i in keep.tolist()]
    return detections


if __name__ == '__main__':
    main()
