"""Bridge to the frozen WildDet3D evaluator; no matching/scoring reimplementation."""
import importlib
import sys
import types
from pathlib import Path

UPSTREAM_COMMIT = '1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a'


def load_official(root):
    """Load evaluation modules without importing unrelated model/depth backends."""
    import subprocess
    root = Path(root).resolve()
    commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(f'Expected evaluator commit {UPSTREAM_COMMIT}, got {commit}')
    for name in ['wilddet3d', 'wilddet3d.data', 'wilddet3d.data.datasets', 'wilddet3d.eval', 'wilddet3d.ops']:
        module = types.ModuleType(name)
        module.__path__ = [str(root.joinpath(*name.split('.')))]
        sys.modules[name] = module
    return importlib.import_module('wilddet3d.eval.detect3d')


def native_to_official_box(center, size_xyz, quaternion_wxyz):
    # Vis4D OPENCV local axes: x=length, y=height, z=width.
    # SpatialEncoder sizes follow local x,y,z. Vis4D stores width,length,height.
    return [*center, size_xyz[2], size_xyz[0], size_xyz[1], *quaternion_wxyz]


def evaluator_kwargs(benchmark, annotation):
    import json
    data = json.loads(Path(annotation).read_text())
    cat_map = {x['name']: x['id'] for x in data['categories']}
    kwargs = dict(annotation=str(annotation), cat_map=cat_map,
                  det_map={x: i for i, x in enumerate(sorted(cat_map))},
                  per_class_eval=True, eval_prox=False, enable_aprel3d=False)
    if benchmark == 'stereo4d':
        kwargs.update(freq_rare_thresh=5, freq_freq_thresh=10)
    elif benchmark == 'in_the_wild':
        kwargs.update(freq_rare_thresh=5, freq_freq_thresh=20)
    elif benchmark == 'scannet':
        from wilddet3d.data.datasets.scannet import scannet_det_map, scannet_class_map
        kwargs.update(det_map=scannet_det_map, cat_map=scannet_class_map,
                      base_classes=['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'toilet', 'sink', 'bathtub'])
    elif benchmark == 'argoverse':
        from wilddet3d.data.datasets.argoverse import av2_det_map, av2_class_map
        kwargs.update(det_map=av2_det_map, cat_map=av2_class_map, eval_prox=True,
                      base_classes=['regular vehicle', 'pedestrian', 'bicyclist', 'construction cone',
                                    'construction barrel', 'large vehicle', 'bus', 'truck',
                                    'vehicular trailer', 'bicycle', 'motorcycle'])
    elif benchmark.startswith('omni3d_'):
        from wilddet3d.data.datasets.omni3d.util import get_dataset_det_map
        from wilddet3d.data.datasets.omni3d.omni3d_classes import omni3d_class_map
        dataset = benchmark.removeprefix('omni3d_')+'_test'
        kwargs.update(det_map=get_dataset_det_map(dataset, omni3d50=True),
                      cat_map=omni3d_class_map, eval_prox=('Objectron' in dataset or 'SUNRGBD' in dataset))
    else:
        raise ValueError(f'Unsupported benchmark: {benchmark}')
    return kwargs
