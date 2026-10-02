"""Download the pinned, public Stereo4D perception validation split (RGB only)."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = 'allenai/WildDet3D-Stereo4D-Bench-Images'
    revision = '25d5ee206aecdfc5469e7f4d55f019fd0432f271'
    paths = {}
    for filename in ['annotations/Stereo4D_val.json', 'packed/images.tar.gz']:
        paths[filename] = hf_hub_download(repo, filename, repo_type='dataset', revision=revision)
        print('Downloaded', filename, flush=True)
    (root/'annotations').mkdir(exist_ok=True)
    shutil.copyfile(paths['annotations/Stereo4D_val.json'], root/'annotations/Stereo4D_val.json')
    with tarfile.open(paths['packed/images.tar.gz']) as archive:
        archive.extractall(root, filter='data')
    data = json.loads((root/'annotations/Stereo4D_val.json').read_text())
    missing = [x['file_path'] for x in data['images'] if not (root/x['file_path']).is_file()]
    status = {'repo': repo, 'revision': revision, 'images': len(data['images']),
              'categories': len(data['categories']), 'annotations': len(data['annotations']),
              'missing_images': missing, 'complete': not missing}
    (root/'download_status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status), flush=True)
    if missing:
        raise RuntimeError('Archive paths do not cover the official annotation images')


if __name__ == '__main__':
    main()
