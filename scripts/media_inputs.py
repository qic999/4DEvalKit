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
    meta = row.get('input_metadata', {})
    if len(paths) != 1:
        views = meta.get('view_ids')
        if not views or len(views) != len(paths) or len(set(views)) != len(views):
            raise ValueError('Multiple videos require ordered, unique view_ids')
        images, frames = [], []
        for path, view in zip(paths, views):
            part = dict(row, media={'video': {'path': path}}, input_metadata={
                k:v for k,v in meta.items() if k != 'view_ids'})
            part_images, part_frames = read_frames(part, video_frames, read_attempts=read_attempts)
            for frame in part_frames:
                frame['camera_id'] = str(view)
                frame['view_id'] = f"camera_{view}/frame_{frame['frame_index']}"
            images.extend(part_images); frames.extend(part_frames)
        return images, frames
    import cv2
    for attempt in range(read_attempts):
        try:
            return _read_video(paths[0], row.get('input_metadata', {}), video_frames)
        except (VideoReadError, OSError, cv2.error) as exc:
            if attempt + 1 == read_attempts:
                logging.getLogger(__name__).warning('Using sequential PTS decoding for %s: %s', paths[0], exc)
                try:
                    return _read_video_pyav(paths[0], row.get('input_metadata', {}), video_frames)
                except Exception as fallback_error:
                    raise VideoReadError(f'Cannot read video with either decoder: {paths[0]}') from fallback_error
            logging.getLogger(__name__).warning('Retrying video read %s (%d/%d): %s',
                                               paths[0], attempt + 1, read_attempts, exc)
            time.sleep(attempt + 1)


def _read_video_pyav(path, meta, video_frames):
    """Decode actual frames/PTS when container counts or random seeking fail.

Some released MotionBench clips declare more frames than exist in the stream.
Scan without retaining images, then decode selected frames in a second pass.
Codec errors still fail the sample; no black frames or question dropping.
"""
    import av
    timestamps = []
    with av.open(str(path)) as container:
        declared = container.streams.video[0].frames
        for frame in container.decode(video=0):
            if frame.time is None or not np.isfinite(frame.time):
                raise VideoReadError('Decoded frame lacks a finite presentation timestamp')
            timestamps.append(float(frame.time))
    if not timestamps or any(b <= a for a,b in zip(timestamps,timestamps[1:])):
        raise VideoReadError('No strictly ordered decoded video timestamps')
    times = np.asarray(timestamps) - timestamps[0]
    start = float(meta.get('time_start') or 0)
    end = float(meta['time_end']) if meta.get('time_end') is not None else times[-1]
    if not np.isfinite(start) or not np.isfinite(end) or start < 0 or end < start:
        raise ValueError('Invalid requested video interval')
    if 'frame_indices' in meta:
        indices = np.asarray(meta['frame_indices'])
        if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype,np.integer)
                or np.any(indices < 0) or np.any(indices >= len(times)) or np.any(np.diff(indices) <= 0)):
            raise ValueError('Explicit frame_indices fall outside decoded video')
    elif start == end:
        if start > times[-1] + 1e-8: raise ValueError('Requested timestamp is outside decoded video')
        indices = np.asarray([int(np.abs(times-start).argmin())])
    else:
        available = np.flatnonzero((times >= start-1e-8) & (times <= end+1e-8))
        if not len(available): raise ValueError('Requested interval is outside decoded video')
        indices = available[np.unique(np.linspace(0,len(available)-1,min(video_frames,len(available))).round().astype(int))]
    selected = set(map(int,indices)); images, frames = [], []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in selected:
                images.append(frame.to_image().convert('RGB'))
                frames.append(dict(path=str(Path(path).resolve()),frame_index=index,timestamp=float(times[index]),
                    view_id=f'frame_{index}',decoding='pyav_actual_frames_pts_v1',
                    container_frame_count=declared,decoded_frame_count=len(times)))
            if index >= indices[-1]: break
    if len(images) != len(indices): raise VideoReadError('Decoded video changed between passes')
    return images,frames


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
        if 'frame_indices' in meta:
            indices = np.asarray(meta['frame_indices'])
            if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= total) or np.any(np.diff(indices) <= 0)):
                raise ValueError('Explicit frame_indices must be increasing, unique, valid integers')
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
