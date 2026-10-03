"""Prepare Physion OCP observations using the official frame_gap=150 schedule.

The fixed OCP cutoff is frame 15 for collisions and frame 45 otherwise. Contact
labels/times never select model observations. --per-scenario makes an explicitly
labeled smoke subset; omit it for the complete released MP4 split.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile

from core.io import source_signature, write_json

URL='https://storage.googleapis.com/physion-dataset/physion_dataset.zip'
SCENARIOS=['collision','drop','towers','link','roll','contain','dominoes']


def scenario(name):
    keys=[k for k in SCENARIOS if k in name and (k!='collision' or 'roll' not in name)]
    if len(keys)!=1:raise ValueError(f'Ambiguous Physion scenario: {name}')
    return keys[0]


def prepare(root, output, loader_source, train_count=None, test_count=None, archive=None):
    import fsspec
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/'status.json',dict(phase='preparing',archive=archive or URL))
    # Read only the literal upstream blacklist; do not import torch/HDF5 loaders.
    tree=ast.parse(Path(loader_source).read_text())
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id=='buggy_stims' for t in n.targets))
    blacklist=assignment.value.func.value.value.split(' ')
    archive_source=archive or URL
    with fsspec.open(archive_source,'rb',block_size=1048576).open() as f:
        with zipfile.ZipFile(f) as z:
            members=set(z.namelist());labels={}
            for split in ['train','test']:
                labels[split]=json.loads(z.read(f'physion_mp4s/{split}/all_video_labels.json'))
    selections={}
    for split,count in [('train',train_count),('test',test_count)]:
        selected=[]
        for kind in SCENARIOS:
            names=sorted(name for name in labels[split] if scenario(name)==kind
                and (split!='test' or name not in blacklist)
                and f'physion_mp4s/{split}/{name}.mp4' in members)
            selected.extend(names[:count])
        selections[split]=selected
    todo=[(split,name) for split,names in selections.items() for name in names]
    def fetch(item):
        split,name=item;member=f'physion_mp4s/{split}/{name}.mp4';path=root/member
        if not path.is_file():
            with fsspec.open(archive_source,'rb',block_size=1048576).open() as f:
                with zipfile.ZipFile(f) as z:data=z.read(member)
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        return str(path.resolve())
    with ThreadPoolExecutor(8) as workers: paths=dict(zip(todo,workers.map(fetch,todo)))
    for split,names in selections.items():
        samples=[];scoring=[]
        for i,name in enumerate(names):
            indices=[0,15] if 'collision' in name and 'roll' not in name else [0,15,30,45]
            sample_id=f'physion_{split}:{i}'
            samples.append(dict(sample_id=sample_id,source_id=name,question='Physical scene observations.',
                media={'video':{'path':paths[split,name]}},input_metadata={'frame_indices':indices}))
            scoring.append(dict(sample_id=sample_id,name=name,scenario=scenario(name),
                label=labels[split][name]['label'],feature_frame_indices=[0,0,0,15] if len(indices)==2 else indices))
        write_json(out/split/'manifest.json',dict(samples=samples,num_samples=len(samples),
            protocol='physion_ocp_frame_gap150_box_features_v1',split=split,
            subset_per_scenario=train_count if split=='train' else test_count,
            upstream_loader=source_signature(loader_source)))
        write_json(out/split/'scoring.json',scoring)
    write_json(out/'status.json',dict(phase='encoder_inputs_ready',
        counts={k:len(v) for k,v in selections.items()},url=URL,
        scope='full' if train_count is None and test_count is None else 'smoke_subset'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--upstream-loader',required=True)
    p.add_argument('--train-per-scenario',type=int);p.add_argument('--test-per-scenario',type=int)
    p.add_argument('--archive',help='Reuse a complete local physion_dataset.zip instead of HTTP ranges')
    a=p.parse_args()
    try:
        prepare(a.data_root,a.output,a.upstream_loader,a.train_per_scenario,a.test_per_scenario,a.archive)
    except Exception as exc:
        write_json(Path(a.output)/'status.json',dict(phase='failed',error=str(exc)))
        raise
