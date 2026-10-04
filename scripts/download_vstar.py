"""Download the pinned public V-STaR videos and verify annotation coverage."""
import argparse
import json
from pathlib import Path
import shutil
import time
import zipfile

from core.io import write_json

REPO = 'V-STaR-Bench/V-STaR'
REVISION = 'd938778037a2c39001039dd54bebdef72fedb2db'
ARCHIVE_BYTES = 29560106192


def download(output):
    from huggingface_hub import hf_hub_download
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = dict(repo=REPO, revision=REVISION, phase='downloading', pid=__import__('os').getpid())

    def update(**kw):
        state.update(kw, updated=time.time())
        write_json(out/'download_status.json', state)

    try:
        # Keep headroom for other users. The archive and extracted videos coexist.
        archive = out/'vstar_videos.zip'
        reserve = 40 * 1024**3
        needed = ARCHIVE_BYTES * (1 if archive.exists() else 2)
        if shutil.disk_usage(out).free < needed + reserve:
            raise OSError('Insufficient disk space for archive, videos and 40 GiB reserve')
        update()
        annotations = hf_hub_download(REPO, 'V_STaR_test.json', repo_type='dataset',
                                     revision=REVISION, local_dir=out)
        archive = Path(hf_hub_download(REPO, archive.name, repo_type='dataset',
                                      revision=REVISION, local_dir=out))
        rows = json.loads(Path(annotations).read_text())
        required = {str(r['vid']) for r in rows}
        with zipfile.ZipFile(archive) as z:
            selected = [i for i in z.infolist() if Path(i.filename).suffix.lower() == '.mp4'
                        and Path(i.filename).stem in required]
            names = [Path(i.filename).stem for i in selected]
            if set(names) != required or len(names) != len(set(names)):
                raise ValueError('Missing or ambiguous annotated videos in archive')
            dest = out/'videos'; dest.mkdir(exist_ok=True)
            update(phase='extracting', total=len(selected), completed=0)
            for i, item in enumerate(selected):
                target = dest/(Path(item.filename).stem+'.mp4')
                # Always stream through ZipExtFile to validate CRC, including resumes.
                part = target.with_suffix('.mp4.part')
                if shutil.disk_usage(out).free < item.file_size + reserve:
                    raise OSError('Extraction reached 40 GiB disk reserve')
                with z.open(item) as src, part.open('wb') as dst:
                    shutil.copyfileobj(src, dst, length=1024*1024)
                part.replace(target)
                update(completed=i+1)
        update(phase='complete', questions=len(rows), videos=len(required))
    except Exception as exc:
        update(phase='failed', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    download(p.parse_args().output)
