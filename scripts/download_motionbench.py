"""Download only labeled MotionBench DEV clips at a pinned dataset revision."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import time

from core.io import write_json

REVISION = 'f099db892172a015c489507c9abe56b036d960ef'


def main():
    from huggingface_hub import HfApi, hf_hub_download
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--workers',type=int,default=8)
    p.add_argument('--limit',type=int)
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    repo='THUDM/MotionBench'
    annotations=hf_hub_download(repo,'MotionBench/video_info.meta.jsonl',repo_type='dataset',revision=REVISION,local_dir=out)
    rows=[json.loads(s) for s in Path(annotations).read_text().splitlines() if s.strip()]
    selected=[r for r in rows if any(q['answer']!='NA' for q in r['qa'])][:a.limit]
    files=HfApi().list_repo_files(repo,repo_type='dataset',revision=REVISION)
    mapping={}
    for f in files:
        if f.startswith('MotionBench/') and f.endswith('.mp4'):
            if Path(f).name in mapping: raise ValueError('Ambiguous clip basename')
            mapping[Path(f).name]=f
    missing=[r['video_path'] for r in selected if r['video_path'] not in mapping]
    if missing: raise FileNotFoundError(f'{len(missing)} DEV clips absent from pinned release: {missing[:5]}')
    targets=sorted({mapping[r['video_path']] for r in selected})
    state=dict(revision=REVISION,total=len(targets),completed=0,errors=[],phase='downloading')
    def fetch(name):
        return hf_hub_download(repo,name,repo_type='dataset',revision=REVISION,local_dir=out)
    with ThreadPoolExecutor(a.workers) as workers:
        futures={workers.submit(fetch,f):f for f in targets}
        for future in as_completed(futures):
            try: future.result();state['completed']+=1
            except Exception as exc:state['errors'].append(dict(file=futures[future],error=str(exc)))
            state['updated']=time.time();write_json(out/'download_status.json',state)
            if state['completed']%100==0:print(state['completed'], '/',len(targets),flush=True)
    state['phase']='failed' if state['errors'] else 'complete';write_json(out/'download_status.json',state)
    if state['errors']:raise RuntimeError('Incomplete DEV download; see download_status.json')


if __name__=='__main__':main()
