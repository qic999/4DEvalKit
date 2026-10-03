"""Numerical fixtures for pinned upstream scorers; these are NOT model scores."""
import argparse
import csv
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

from core.io import write_json
from core.official_4d import adt_evaluator, tapvid_evaluator, vstar_metrics


def verify(output, cache=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    header=['timestamp_ns','prototype','t_wo_x','t_wo_y','t_wo_z','q_wo_x','q_wo_y','q_wo_z','q_wo_w']
    def poses(offset):
        f=io.StringIO();w=csv.writer(f);w.writerow(header);w.writerow([0,'cube',offset,0,0,0,0,0,1]);return f.getvalue()
    gt=out/'fixture_gt.zip';good=out/'fixture_good.zip';bad=out/'fixture_bad.zip'
    with zipfile.ZipFile(gt,'w') as z:
        z.writestr('sequence.csv',poses(0))
        z.writestr('vertices.json',json.dumps({'cube':list(itertools.product([-.5,.5],repeat=3))}))
        z.writestr('symmetries.json',json.dumps({'cube':[]}))
        z.writestr('diameters.json',json.dumps({'cube':3**.5}))
        f=io.StringIO();w=csv.writer(f)
        w.writerow(['prototype']+[f'p_local_obj_{axis}{side}[m]' for axis in 'xyz' for side in ['min','max']])
        w.writerow(['cube']+[-.5,.5]*3);z.writestr('3d_bounding_box.csv',f.getvalue())
    for file,offset in [(good,0),(bad,3)]:
        with zipfile.ZipFile(file,'w') as z:z.writestr('sequence.csv',poses(offset))
    fn=adt_evaluator(cache)
    assert fn(gt,good,['sequence'],['cube'])['result'][0]['test_split']['mAP']==1
    assert fn(gt,bad,['sequence'],['cube'])['result'][0]['test_split']['mAP']==0
    xyz=np.zeros((2,4,3));xyz[...,2]=2;occ=np.zeros((2,4),dtype=bool)
    fn=tapvid_evaluator(cache)
    good_metrics=fn(occ,xyz,occ,xyz,np.array([100,100,50,50]),scaling='none')
    bad_metrics=fn(occ,xyz,occ,xyz+10,np.array([100,100,50,50]),scaling='none')
    assert np.allclose(good_metrics['average_jaccard'],1)
    assert np.allclose(bad_metrics['average_jaccard'],0)
    f=vstar_metrics(cache)
    assert f['calculate_temporal_iou']([0,2],[0,2])==1
    assert f['calculate_temporal_iou']([0,2],[3,4])==0
    gt_boxes=[dict(timestamp=0,xmin=0,ymin=0,xmax=10,ymax=10)]
    assert f['calculate_spatial_metrics'](gt_boxes,{'0':[0,0,10,10]})[1]==1
    assert f['calculate_spatial_metrics'](gt_boxes,{})[1]==0
    report=dict(purpose='synthetic_metric_fixtures_not_model_evaluation',
        checks={'ADT':'perfect and displaced poses passed','TAPVid-3D':'perfect and displaced points passed',
                'V-STaR-grounding':'perfect, disjoint, and missing predictions passed'})
    write_json(out/'verification.json',report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--cache')
    a=p.parse_args();print(json.dumps(verify(a.output,a.cache),indent=2))
