"""Read public benchmark media in source order, retaining real video timing."""
from pathlib import Path
import logging
import time
import numpy as np
from PIL import Image


def flatten_media(value):
    if isinstance(value, list):
        return [p for item in value for p in flatten_media(item)]
    if isinstance(value, dict) and value.get('path'):
        return [value['path']]
    raise ValueError('Media must be exported to local files first')


class VideoReadError(RuntimeError):
    """A video could not be opened or decoded, possibly due to transient I/O."""


def read_frames(row, video_frames=16, *, read_attempts=3):
    if video_frames < 1 or read_attempts < 1:
        raise ValueError('Frame count and read attempts must be positive')
    media = row['media']
    if 'video' not in media:
        paths = flatten_media(media['image'])
        return [Image.open(p).convert('RGB') for p in paths], [
            {'path': str(Path(p).resolve()), 'view_id': f'view_{i}'} for i,p in enumerate(paths)]
    paths = flatten_media(media['video'])
    if len(paths) != 1:
        raise ValueError('Expected one encoded video per question')
    import cv2
    for attempt in range(read_attempts):
        try:
            return _read_video(paths[0], row.get('input_metadata', {}), video_frames)
        except (VideoReadError, OSError, cv2.error) as exc:
            if attempt + 1 == read_attempts:
                raise VideoReadError(f'Cannot read video after {read_attempts} attempts: {paths[0]}') from exc
            logging.getLogger(__name__).warning('Retrying video read %s (%d/%d): %s',
                                               paths[0], attempt + 1, read_attempts, exc)
            time.sleep(attempt + 1)


def _read_video(path, meta, video_frames):
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        count, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(count) or not np.isfinite(fps) or count < 1 or fps <= 0:
            raise VideoReadError('Missing video frame count or fps')
        total = int(count)
        start_time = float(meta.get('time_start') or 0)
        end_time = meta.get('time_end')
        if not np.isfinite(start_time) or end_time is not None and not np.isfinite(float(end_time)):
            raise ValueError('Video times must be finite')
        if end_time is not None and start_time == float(end_time):
            # Point annotations use the nearest frame. A timestamp equal to the
            # container duration refers to the final frame, not a nonexistent
            # frame at index `total`. Keep its actual timestamp in the output.
            if start_time < 0 or start_time > total/fps + 1e-9:
                raise ValueError('Requested time interval is outside the video')
            start = stop = min(total-1, int(np.floor(start_time*fps + .5)))
        else:
            start = max(0, int(np.ceil(start_time * fps)))
            stop = total-1 if end_time is None else min(total-1, int(np.floor(float(end_time)*fps)))
        if stop < start:
            raise ValueError('Requested time interval is outside the video')
        indices = np.unique(np.linspace(start, stop, min(video_frames, stop-start+1)).round().astype(int))
        images, frames = [], []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise VideoReadError(f'Cannot decode video frame {index}: {path}')
            images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            frames.append({'path': str(Path(path).resolve()), 'frame_index': int(index),
                           'timestamp': float(index/fps), 'view_id': f'frame_{int(index)}'})
        return images, frames
    finally:
        cap.release()
