"""Encode train/test OCP clips, pool predicted boxes, run the official readout."""
import argparse
from pathlib import Path
import subprocess
import sys

from core.io import read_json, write_json
from scripts.physion_box_features import export


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--upstream-cache')
    a=p.parse_args();c=read_json(a.config);out=Path(a.output).resolve()
    if c.get('modes')!=[]:raise ValueError('Physion feature config must have modes=[]')
    subprocess.run([sys.executable,'-m','scripts.run_additional_4d','--config',a.config,'--output',str(out)],check=True)
    if read_json(out/'suite.json')['phase']!='complete':raise RuntimeError('Incomplete Physion geometry')
    jobs={j['benchmark']:j for j in c['jobs']}
    for split in ['train','test']:
        name='Physion-'+split
        export(jobs[name]['prepared'],out/name/'geometry/geometry.json',out/'features'/f'{split}.hdf5')
    config=dict(train_path=str(out/'features/train.hdf5'),test_path=str(out/'features/test.hdf5'),
        train_scenario_indices=str(out/'features/train.indices.json'),
        test_scenario_indices=str(out/'features/test.indices.json'),
        test_scenario_map=str(out/'features/test.mapping.json'),model_name='wds518_e100_ocp_box_features')
    write_json(out/'readout_config.json',config)
    command=[sys.executable,'-m','scripts.score_official_4d','--benchmark','Physion',
             '--config',str(out/'readout_config.json'),'--output',str(out/'readout.json')]
    if a.upstream_cache:command+=['--cache',a.upstream_cache]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
