"""Recover original benchmark RGB using COCO and the Objects365 source archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time

from core.io import read_json, write_json


def main():
    from huggingface_hub import hf_hub_download, HfApi, hf_hub_url
    import pyarrow.parquet as pq
    import requests
    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--annotation', required=True)
    p.add_argument('--scorer-python', required=True)
    p.add_argument('--follow-runner', action='store_true')
    args = p.parse_args()
    root = Path(args.run_root).resolve()
    output = root/'data/in_the_wild'
    (output/'annotations').mkdir(parents=True, exist_ok=True)
    annotation = output/'annotations/InTheWild_v3_val.json'
    shutil.copyfile(args.annotation, annotation)
    data = read_json(annotation)
    images = {i['file_path']:i for i in data['images']}
    status_path = root/'in_the_wild_download_status.json'

    def save_rgb(name, content):
        im = Image.open(io.BytesIO(content))
        expected = images[name]
        if im.size != (expected['width'], expected['height']):
            raise ValueError(f'Original source dimensions differ for {name}: {im.size}')
        im.verify()
        target = output/name
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp')
        temp.write_bytes(content)
        temp.replace(target)

    def coco(name):
        if (output/name).exists():
            return
        split = 'train2017' if '/coco_train/' in name else 'val2017'
        url = f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{Path(name).name}'
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=90)
                response.raise_for_status()
                save_rgb(name, response.content)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    try:
        write_json(status_path, {'state':'downloading COCO RGB', 'complete':False})
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(coco, [n for n in images if '/coco_' in n]))
        print('COCO source RGB ready', flush=True)
        needed = {int(Path(n).stem.split('_')[-1]):n for n in images if '/obj365_val/' in n}
        repo, revision = 'jxu124/objects365', 'b9ce8d3596a3642fb909226750017d75dbc669a5'
        files = [s.rfilename for s in HfApi().dataset_info(repo, revision=revision).siblings if s.rfilename.endswith('.parquet')]
        plans, found = {}, set()
        for file in files:
            path = hf_hub_download(repo, file, repo_type='dataset', revision=revision)
            table = pq.read_table(path, columns=['image_info'])
            split = 'val' if 'validation-' in file else 'train'
            for batch in table.to_batches(max_chunksize=20000):
                for row in batch.to_pylist():
                    im = row['image_info']
                    if im['id'] not in needed:
                        continue
                    target = needed[im['id']]
                    if (im['width'],im['height']) != (images[target]['width'],images[target]['height']):
                        raise ValueError('Objects365 image ID matched but source dimensions differ')
                    if im['id'] in found:
                        raise ValueError('Duplicate source image ID across Objects365 splits')
                    found.add(im['id'])
                    patch = re.search(r'patch\d+', im['file_name']).group(0)
                    plans.setdefault(split+'/'+patch, {})[Path(im['file_name']).name] = target
            print('Mapped original Objects365 IDs', len(found), '/', len(needed), flush=True)
        # Some old v1 validation images moved to the v2 test partition, whose
        # annotations are hidden. Recover exact filename matches from its public
        # RGB archives; check original dimensions before accepting any image.
        missing_ids = sorted(set(needed)-found)
        if missing_ids:
            targets = {f'objects365_v1_{i:08d}.jpg':needed[i] for i in missing_ids}
            for patch in range(16):
                plans[f'test/patch{patch}'] = targets
            print('Looking for',len(missing_ids),'v1 IDs in public v2 test RGB archives',flush=True)
        write_json(root/'objects365_source_archive_plan.json', {'index_repo':repo, 'index_revision':revision,'archives':plans})
        write_json(status_path, {'state':'downloading Objects365 source archives', 'complete':False, 'archives':len(plans)})

        def archive_job(job):
            key, targets = job
            if all((output/n).exists() for n in targets.values()):
                return
            split, patch = key.split('/')
            if split == 'train':
                # Mirror of the original archive, not resized or annotated images.
                url = hf_hub_url('guozonghao96/objects365', patch+'.tar.gz', repo_type='dataset',
                                 revision='ae34a8ecaed566c69b637bd69b91daf0a95157af')
            else:
                url = ('https://dorc.ks3-cn-beijing.ksyun.com/data-set/'
                       '2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/'+split+'/images/v1/'+patch+'.tar.gz')
            for attempt in range(3):
                try:
                    print('Streaming', key, len(targets), 'requested images', flush=True)
                    with requests.get(url, stream=True, timeout=180) as response:
                        response.raise_for_status()
                        with tarfile.open(fileobj=response.raw, mode='r|gz') as archive:
                            for member in archive:
                                name = Path(member.name).name
                                if name in targets and member.isfile():
                                    save_rgb(targets[name], archive.extractfile(member).read())
                    if split != 'test' and not all((output/n).exists() for n in targets.values()):
                        raise ValueError(f'Archive {key} does not contain all mapped original images')
                    print('READY', key, flush=True)
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(5)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(archive_job, job) for job in plans.items()]):
                future.result()
        missing_rgb = [n for n in images if not (output/n).exists()]
        if missing_rgb:
            write_json(root/'in_the_wild_missing_rgb.json', missing_rgb)
            raise ValueError(f'{len(missing_rgb)} original RGB images unavailable after checking all source archives')
        manifest = root/'in_the_wild_manifest.json'
        subprocess.run([args.scorer_python,'-m','scripts.prepare_wilddet3d_perception',
            '--upstream',str(root/'upstream'),'--annotation',str(annotation),'--data-root',str(output),
            '--benchmark','in_the_wild','--output',str(manifest)],check=True)
        write_json(status_path, {'state':'ready', 'complete':True,'images':len(images)})
        if args.follow_runner:
            identity = read_json(root/'runner_process.json')
            while True:
                try:
                    same = Path(f'/proc/{identity["pid"]}/stat').read_text().split()[21] == identity['start_ticks']
                except FileNotFoundError:
                    same = False
                if not same:
                    break
                time.sleep(30)
            cfg = read_json(root/'run_config.json')
            cfg['datasets'] = [{'name':'in_the_wild','manifest':str(manifest)}]
            cfg_path = root/'in_the_wild_run_config.json'
            write_json(cfg_path, cfg)
            subprocess.run([str(Path('.venv/bin/python').absolute()),'-m','scripts.run_wilddet3d_perception',
                            '--config',str(cfg_path)],check=True)
    except Exception as error:
        write_json(status_path, {'state':'failed', 'complete':False, 'error':f'{type(error).__name__}: {error}',
                                'ready_images':sum((output/n).exists() for n in images), 'total_images':len(images)})
        raise


if __name__ == '__main__':
    main()
