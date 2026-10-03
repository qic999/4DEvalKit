"""Pool predicted per-frame boxes into fixed OCP features for the official probe.

This measures the box representation, not an unspecified encoder hidden layer.
All feature normalization and readout fitting happen on readout training data.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from core.io import read_json, write_json
from core.geometry import GeometryStore


def box_features(scene, frame_indices):
    frames=defaultdict(list)
    for track in scene.get('tracks',[]):
        for obs in track['observations']:
            index=int(obs['view_id'].split('frame_')[-1])
            frames[index].append(obs['center']+obs['size']+obs.get('quaternion_xyzw',[0,0,0,1])+[obs.get('confidence',0)])
    output=[]
    for index in frame_indices:
        values=np.asarray(frames[index],dtype=np.float32)
        if len(values):
            output.append(np.concatenate([values.mean(0),values.std(0),values.min(0),values.max(0),[len(values)]]))
        else:output.append(np.zeros(45))
    return np.asarray(output,dtype=np.float32)


def export(prepared, geometry, output):
    import h5py
    rows=read_json(Path(prepared)/'scoring.json');scenes=GeometryStore(geometry).scenes
    if set(scenes)!=set(r['sample_id'] for r in rows):raise ValueError('Incomplete Physion geometry')
    indices=defaultdict(list);mapping={}
    for i,row in enumerate(rows):indices[row['scenario']].append(i);mapping[row['name']+'.hdf5']=i
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(out,'w') as f:
        f['features']=np.stack([box_features(scenes[r['sample_id']],r['feature_frame_indices']) for r in rows])
        f['label']=np.asarray([r['label'] for r in rows])
    write_json(out.with_suffix('.indices.json'),dict(indices));write_json(out.with_suffix('.mapping.json'),mapping)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True)
    p.add_argument('--geometry',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();export(a.prepared,a.geometry,a.output)
