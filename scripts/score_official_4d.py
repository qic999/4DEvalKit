"""Run pinned official ADT/TAPVid-3D/V-STaR geometry/Physion numerical scorers.

These entry points consume native predictions. They do not fabricate pose,
point, visibility, grounding or feature predictions from object boxes.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from core.io import write_json
from core.official_4d import adt_evaluator, tapvid_evaluator, vstar_metrics, physion_readout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['ADT','TAPVid-3D','V-STaR-grounding','Physion'])
    p.add_argument('--config',required=True,help='JSON with native input paths and protocol options')
    p.add_argument('--output',required=True)
    p.add_argument('--cache')
    a=p.parse_args(); c=json.loads(Path(a.config).read_text())
    if a.benchmark=='ADT':
        stats=adt_evaluator(a.cache)(c['annotations'],c['predictions'],c['sequences'],c['prototypes'])
    elif a.benchmark=='TAPVid-3D':
        import numpy as np
        gt=np.load(c['annotations'],allow_pickle=False);pred=np.load(c['predictions'],allow_pickle=False)
        stats=tapvid_evaluator(a.cache)(gt['occluded'],gt['tracks'],pred['occluded'],pred['tracks'],
            gt['intrinsics_params'],scaling=c['scaling'],order=c.get('order','n t'),
            query_points=gt['query_points'] if 'query_points' in gt.files else None)
        stats={k:np.asarray(v).tolist() for k,v in stats.items()}
    elif a.benchmark=='V-STaR-grounding':
        import numpy as np
        gt=json.loads(Path(c['annotations']).read_text());pred=json.loads(Path(c['predictions']).read_text())
        if set(pred)!=set(str(i) for i in range(len(gt))):raise ValueError('Expected predictions for every annotation index')
        f=vstar_metrics(a.cache); rows=[]
        for i,row in enumerate(gt):
            answer=pred[str(i)]
            aps,miou=f['calculate_spatial_metrics'](row['bboxes'],answer.get('spatial',{}))
            rows.append(dict(temporal_iou=f['calculate_temporal_iou'](row['timestamps'],answer.get('temporal')),
                             spatial_iou=float(miou),spatial_ap=[float(x) for x in aps]))
        stats=dict(temporal_iou=float(np.mean([r['temporal_iou'] for r in rows])),
                   spatial_iou=float(np.mean([r['spatial_iou'] for r in rows])),samples=len(rows),
                   limitation='Grounding diagnostics only; no 72B semantic judge or joint official score',results=rows)
    else:
        out=Path(a.output).resolve().parent/'physion';out.mkdir(parents=True,exist_ok=True)
        defaults=dict(clf_C=[1e-6,1e-5,.01,.1,1,5,10,20],random_state=42,model_name='spatial_box_features',save_path=str(out))
        defaults.update(c);physion_readout(a.cache).train(SimpleNamespace(**defaults))
        stats=json.loads((Path(defaults['save_path'])/(defaults['model_name']+'_results.json')).read_text())
    write_json(a.output,dict(benchmark=a.benchmark,protocol='pinned_upstream_numerical_functions',config=c,statistics=stats))


if __name__=='__main__':main()
