"""Recover only official Omni3D test RGB from an existing user's raw-data archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

from core.io import read_json, write_json


def main():
    from huggingface_hub import hf_hub_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--inventory-root', required=True)
    p.add_argument('--scorer-python', required=True)
    p.add_argument('--archive-repo', required=True)
    p.add_argument('--archive-revision', required=True)
    args = p.parse_args()
    root = Path(args.run_root).resolve()
    data_root = root/'data/omni3d'
    inventory = Path(args.inventory_root)
    needed, datasets = set(), {}
    for path in (data_root/'annotations').glob('*_test.json'):
        names = {str(Path('datasets')/im['file_path']) for im in read_json(path)['images']}
        needed |= names
        datasets[path.stem] = names
    jobs, covered = [], set()
    state = read_json(inventory.parent/'state.json')
    for path in sorted(inventory.glob('*.files0')):
        names = {str(Path(x.decode())) for x in path.read_bytes().split(b'\0') if x}
        wanted = names & needed
        if wanted:
            index = int(path.stem.split('-')[-1])
            jobs.append((state['shards'][index], wanted))
            covered |= wanted
    missing = sorted(needed-covered)
    write_json(root/'omni_download_plan.json', {'repo': args.archive_repo, 'revision': args.archive_revision,
        'datasets': {k: len(v) for k,v in datasets.items()}, 'missing_inventory': missing,
        'shards': [{'remote_path': j['remote_path'], 'images': len(names)} for j,names in jobs]})
    if missing:
        raise ValueError(f'{len(missing)} official test images are absent from archive inventory')

    def publish_ready():
        for name, paths in datasets.items():
            manifest = root/f'omni3d_{name.removesuffix("_test")}_manifest.json'
            if manifest.exists() or not all((data_root/Path(p).relative_to('datasets')).is_file() for p in paths):
                continue
            temporary = manifest.with_suffix('.preparing.json')
            subprocess.run([args.scorer_python, '-m', 'scripts.prepare_wilddet3d_perception',
                '--upstream', str(root/'upstream'), '--annotation', str(data_root/'annotations'/f'{name}.json'),
                '--data-root', str(data_root), '--benchmark', 'omni3d_'+name.removesuffix('_test'),
                '--output', str(temporary)], check=True)
            temporary.replace(manifest)
            print('READY', name, flush=True)

    def fetch(job):
        shard, wanted = job
        if all((data_root/Path(p).relative_to('datasets')).is_file() for p in wanted):
            return
        print('Downloading', shard['remote_path'], len(wanted), 'test images', flush=True)
        archive = Path(hf_hub_download(args.archive_repo, shard['remote_path'], repo_type='dataset',
            revision=args.archive_revision, local_dir=root/'temporary_archives'))
        checksum = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
        if checksum != shard['sha256']:
            raise ValueError('Archive checksum differs from upload manifest')
        extracted = set()
        with tarfile.open(archive) as tar:
            for member in tar:
                name = str(Path(member.name))
                if name in wanted:
                    if not member.isfile():
                        raise ValueError('Expected RGB regular file')
                    dest = data_root/Path(name).relative_to('datasets')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_suffix(dest.suffix+'.tmp')
                    temp.write_bytes(tar.extractfile(member).read())
                    temp.replace(dest)
                    extracted.add(name)
        if wanted != extracted:
            raise ValueError('Archive did not contain all indexed test images')
        write_json(root/'data/omni3d'/f'extracted_{archive.stem}.json',
                   {'source': shard['remote_path'], 'sha256': checksum, 'images': len(extracted)})
        archive.unlink()  # Only the transient archive this task downloaded; keep extracted RGB.
        print('Extracted', shard['remote_path'], len(extracted), flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            future.result()
            publish_ready()
    publish_ready()
    write_json(root/'omni_download_status.json', {'complete': True, 'images': len(needed)})


if __name__ == '__main__':
    main()
