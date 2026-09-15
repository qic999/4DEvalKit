"""Export a ScanNet GT-object oracle in the native encoder's world coordinates."""
import argparse
from pathlib import Path

from benchmark.loader import BenchmarkSession, media_manifest
from core.geometry import normalize_scene
from core.io import read_json, write_json, source_signature
from scripts.benchmark_registry import get_spec


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scannet-root',required=True)
    p.add_argument('--qa-json',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation
    root=Path(args.scannet_root); out=Path(args.output)
    metadata=read_json(root/'vsi_scannet_meta.json')
    translations=read_json(root/'vsi_scannet_translation.json')
    session=BenchmarkSession(get_spec('VSI-Bench'),data=args.qa_json,dataset_filter='scannet',
                             model_name='qwen35_9b',seed=0,batch_size=16)
    public=[media_manifest(e) for batch in session.batches() for e in batch]
    scenes={};validation=[]
    for name in sorted({r['input_metadata']['scene_name'] for r in public}):
        objects=[]
        for category,boxes in metadata[name]['object_bbox'].items():
            for i,box in enumerate(boxes):
                # ScanNet stores basis vectors consecutively; transpose them
                # into the column-vector convention used by native camera_R.
                matrix=np.asarray(box['normalizedAxes']).reshape(3,3).T
                if not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-4):
                    raise ValueError(f'Nonorthogonal GT axes in {name}')
                if np.linalg.det(matrix)<0:
                    matrix[:,2]*=-1  # An OBB is unchanged by an axis sign flip.
                center=np.asarray(box['centroid'])+translations[name]
                objects.append({'instance_id':f'{category}_{i}', 'category':f'{category}_{i}',
                    'center':center.tolist(), 'size':box['axesLengths'],
                    'quaternion_xyzw':Rotation.from_matrix(matrix).as_quat().tolist()})
        scenes[name]=normalize_scene({'units':'m','coordinate_frame':'scannet_world_z_up',
            'objects':objects,'provenance':{'geometry_source':'oracle','label_source':'gt_category',
                'object_coverage':'all annotated scene objects',
                'source':source_signature(root/'vsi_scannet_meta.json'),
                'translation_source':source_signature(root/'vsi_scannet_translation.json')}})
    # Cross-check a source-frame annotation against the exported world box.
    name=next(iter(scenes));native=read_json(root/'val-json'/f'{name}.json')
    for obj in scenes[name]['objects']:
        observations=native['objects'].get(obj['instance_id'],{})
        if not observations:continue
        frame,record=next(iter(observations.items()));pose=np.asarray(native['frames'][frame]['camera_pose'])
        center=pose[:3,:3]@record['camera_center']+pose[:3,3]
        error=float(np.max(np.abs(center-obj['center'])))
        if error>1e-4:raise ValueError('GT oracle world frame does not match native frame annotations')
        validation.append({'scene':name,'object':obj['instance_id'],'center_max_error_m':error})
    if not validation:raise ValueError('No native-frame oracle validation was possible')
    write_json(out/'geometry.json',{'scenes':scenes})
    write_json(out/'manifest.json',{'benchmark':'VSI-Bench','samples':public})
    write_json(out/'validation.json',{'scenes':len(scenes),'questions':len(public),'checks':validation,
        'limitations':'Oracle uses all GT objects and categories; excludes room_size, object_counts and QA answers. RGB arms are evaluated separately on matched RGB benchmark inputs.'})
    print(f'Exported {len(scenes)} oracle scenes and {len(public)} questions')


if __name__=='__main__':main()
