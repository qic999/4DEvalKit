"""Launch the official nuScenes Tracking or HOT3D BOP evaluator unchanged.

The config supplies native predictions, dataset root, pinned upstream checkout,
and that checkout's Python environment. No training/test labels are inferred.
"""
import argparse
import os
from pathlib import Path
import subprocess

from core.io import read_json, write_json

PINS={'nuScenes-Tracking':'b40adc467b919192899405d9b77871afee8efa07',
      'HOT3D-BOP':'cea62d651c7e395b2e1962b9749e4e89693c6ac4'}


def command(benchmark,c,output):
    repo=Path(c['upstream_repo']).resolve();python=c['python'];out=Path(output).resolve()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if revision!=PINS[benchmark]:raise ValueError(f'Expected upstream revision {PINS[benchmark]}, got {revision}')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=repo,text=True).strip():
        raise ValueError('Upstream tracked source has local changes')
    if not Path(c['data_root']).is_dir():raise FileNotFoundError(c['data_root'])
    env=dict(os.environ)
    if benchmark=='nuScenes-Tracking':
        entry=repo/'python-sdk/nuscenes/eval/tracking/evaluate.py'
        env['PYTHONPATH']=str(repo/'python-sdk')+os.pathsep+env.get('PYTHONPATH','')
        cmd=[python,str(entry),str(Path(c['predictions']).resolve()),'--dataroot',str(Path(c['data_root']).resolve()),
             '--version',c.get('version','v1.0-trainval'),'--eval_set',c.get('split','val'),'--output_dir',str(out)]
    else:
        entry=repo/'scripts/eval_bop24_pose.py'
        env['PYTHONPATH']=str(repo)+os.pathsep+env.get('PYTHONPATH','')
        env['BOP_PATH']=str(Path(c['data_root']).resolve())
        predictions=Path(c['predictions']).resolve()
        cmd=[python,str(entry),'--result_filenames',predictions.name,'--results_path',str(predictions.parent),
             '--eval_path',str(out),'--targets_filename',c['targets_filename']]
    return cmd,env,repo


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',choices=list(PINS),required=True)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();c=read_json(a.config)
    cmd,env,cwd=command(a.benchmark,c,a.output)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'launch.json',dict(benchmark=a.benchmark,upstream_revision=PINS[a.benchmark],command=cmd,dry_run=a.dry_run))
    if a.dry_run:print(cmd);return
    with (out/'official_eval.log').open('ab') as log:
        subprocess.run(cmd,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


if __name__=='__main__':main()
