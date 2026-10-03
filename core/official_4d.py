"""Execute audited, hash-verified upstream metric functions without model imports.

Sources remain unmodified in an ignored cache. Selecting function definitions
avoids unrelated C++ visualization imports (ADT) and the eager 72B judge load
(V-STaR). Numerical function bodies are compiled verbatim, never rewritten.
"""
import ast
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def source(repo, path, cache=None):
    manifest=json.loads((ROOT/'reports/upstream_eval_audit_20261002/manifest.json').read_text())
    entries=next(v for v in manifest.values() if isinstance(v,list) and v and isinstance(v[0],dict)
                 and 'sha256' in v[0] and 'url' in v[0])
    record=next(r for r in entries if r['repo']==repo and r['path']==path)
    dest=Path(cache or ROOT/'external/official_4d')/repo/path
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        data=urllib.request.urlopen(record['url'],timeout=60).read()
        if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Upstream source hash mismatch')
        dest.write_bytes(data)
    if hashlib.sha256(dest.read_bytes()).hexdigest()!=record['sha256']:
        raise ValueError(f'Changed upstream metric source: {dest}')
    return dest


def functions(path, names, namespace):
    tree=ast.parse(Path(path).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise ValueError('Missing upstream metric functions')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def adt_evaluator(cache=None):
    import csv, io, zipfile
    import numpy as np
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm
    from typing import Dict,List,Optional
    ns=dict(csv=csv,io=io,json=json,zipfile=zipfile,np=np,_np=np,R=Rotation,tqdm=tqdm,
            Dict=Dict,List=List,Optional=Optional,MIN_SYM_ANGLE_STEP=.01,
            THRESHOLDS=[.05,.10,.15,.20,.25,.30,.35,.40,.45,.50])
    functions(source('facebookresearch/projectaria_tools','projectaria_tools/projects/adt/utils.py',cache),
        ['get_rotation_matrices','apply_pose','compute_mssd','voc_ap','get_timed_poses','get_vertices',
         'get_3d_bounding_box','get_symmetries','get_diameters'],ns)
    functions(source('facebookresearch/projectaria_tools',
        'projects/AriaDigitalTwinDatasetTools/challenges/evaluate.py',cache),['evaluate'],ns)
    return ns['evaluate']


def tapvid_evaluator(cache=None):
    import importlib.util
    path=source('google-deepmind/tapnet','tapnet/tapvid3d/evaluation/metrics.py',cache)
    spec=importlib.util.spec_from_file_location('official_tapvid3d_metrics',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.compute_tapvid3d_metrics


def vstar_metrics(cache=None):
    import numpy as np
    return functions(source('V-STaR-Bench/V-STaR','eval.py',cache),
        ['calculate_temporal_iou','compute_iou','calculate_bbox_iou','calculate_spatial_metrics'],
        dict(np=np,ast=ast))


def physion_readout(cache=None):
    import importlib.util
    path=source('neuroailab/physion_evaluator','physion_evaluator/train_readout.py',cache)
    spec=importlib.util.spec_from_file_location('official_physion_readout',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
