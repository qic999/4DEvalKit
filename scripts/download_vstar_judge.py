"""Fetch the exact official V-STaR judge model with an explicit disk reserve."""
import argparse
import os
from pathlib import Path
import shutil
import time

from core.io import write_json

REPO = 'Qwen/Qwen2.5-72B-Instruct'
REVISION = '495f39366efef23836d0cfae4fbe635880d2be31'


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    a = p.parse_args(); out = Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    state = dict(repo=REPO, revision=REVISION, pid=os.getpid(), phase='downloading')

    def update(**kw):
        state.update(kw, updated=time.time()); write_json(out/'download_status.json', state)

    try:
        update()
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
        names = [f.rfilename for f in info.siblings if f.rfilename.endswith(('.json', '.safetensors', '.txt'))]
        sizes = {f.rfilename: f.size for f in info.siblings if f.rfilename in names}
        remaining = sum(n for f, n in sizes.items() if not (out/f).exists() or (out/f).stat().st_size != n)
        # Leave space for V-STaR extraction and unrelated work during this download.
        if shutil.disk_usage(out).free < remaining + 75*1024**3:
            raise OSError('Judge weights plus 75 GiB reserve exceed available disk space')
        update(total_bytes=sum(sizes.values()))
        snapshot_download(REPO, revision=REVISION, local_dir=out, allow_patterns=names, max_workers=3)
        if any(not (out/f).is_file() or (out/f).stat().st_size != n for f, n in sizes.items()):
            raise ValueError('Incomplete judge snapshot')
        update(phase='complete')
    except Exception as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}'); raise


if __name__ == '__main__':
    main()
