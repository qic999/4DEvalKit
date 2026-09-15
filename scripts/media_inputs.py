"""Read public benchmark media in source order, retaining real video timing."""
from pathlib import Path
import numpy as np
from PIL import Image


def flatten_media(value):
    if isinstance(value, list):
        return [p for item in value for p in flatten_media(item)]
    if isinstance(value, dict) and value.get('path'):
        return [value['path']]
    raise ValueError('Media must be exported to local files first')


def read_frames(row, video_frames=16):
    media = row['media']
    if 'video' not in media:
        paths = flatten_media(media['image'])
        return [Image.open(p).convert('RGB') for p in paths], [
            {'path': str(Path(p).resolve()), 'view_id': f'view_{i}'} for i,p in enumerate(paths)]
    paths = flatten_media(media['video'])
    if len(paths) != 1:
        raise ValueError('Expected one encoded video per question')
    import cv2
    cap = cv2.VideoCapture(paths[0])
    try:
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or fps <= 0:
            raise ValueError('Missing video frame count or fps')
        meta = row.get('input_metadata', {})
        start = max(0, int(np.ceil(float(meta.get('time_start') or 0) * fps)))
        end_time = meta.get('time_end')
        stop = total-1 if end_time is None else min(total-1, int(np.floor(float(end_time)*fps)))
        if stop < start:
            raise ValueError('Requested time interval is outside the video')
        indices = np.unique(np.linspace(start, stop, min(video_frames, stop-start+1)).round().astype(int))
        images, frames = [], []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode video frame {index}: {paths[0]}')
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append({'path': str(Path(paths[0]).resolve()), 'frame_index': int(index),
                           'timestamp': float(index/fps), 'view_id': f'frame_{int(index)}'})
        return images, frames
    finally:
        cap.release()
