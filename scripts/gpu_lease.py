"""Coordinate toolkit reasoning servers on a GPU without stopping other jobs."""
import fcntl
import os
from pathlib import Path
import time


def acquire_gpu(gpu, on_wait=None):
    path = Path('/tmp') / f'4devalkit-{os.getuid()}-gpu-{int(gpu)}.lock'
    stream = path.open('a+')
    try:
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return stream
            except BlockingIOError:
                if on_wait:
                    on_wait()
                time.sleep(5)
    except BaseException:
        stream.close()
        raise
