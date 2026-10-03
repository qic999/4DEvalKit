"""Prepare Physion OCP observations using the official frame_gap=150 schedule.

The fixed OCP cutoff is frame 15 for collisions and frame 45 otherwise. Contact
labels/times never select model observations. --per-scenario makes an explicitly
labeled smoke subset; omit it for the complete released MP4 split.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import json
from itertools import islice
from pathlib import Path
import zipfile

from core.io import source_signature, write_json

URL='https://storage.googleapis.com/physion-dataset/physion_dataset.zip'
SCENARIOS=['collision','drop','towers','link','roll','contain','dominoes']


def scenario(name):
    keys=[k for k in SCENARIOS if k in name and (k!='collision' or 'roll' not in name)]
    if len(keys)!=1:raise ValueError(f'Ambiguous Physion scenario: {name}')
    return keys[0]


def ocp_frame_indices(name, num_frames):
    """Match the official MP4 loader's clip-to-last-frame and left padding."""
    if num_frames < 1:
        raise ValueError('Physion video contains no decoded frames')
    requested = [0, 0, 0, 15] if 'collision' in name and 'roll' not in name else [0, 15, 30, 45]
    return [min(index, num_frames - 1) for index in requested]


def decoded_ocp_schedule(path, name):
    import av
    # Only the public OCP prefix is needed. Counting up to its cutoff produces
    # the same clipped indices as counting the entire video, without later GT.
    cutoff = 15 if 'collision' in name and 'roll' not in name else 45
    with av.open(str(path)) as container:
        count = sum(1 for _ in islice(container.decode(video=0), cutoff + 1))
    return ocp_frame_indices(name, count)


def refresh_frame_schedule(output, workers=8):
    """Repair prepared schedules; retain IDs, labels and unaffected cache inputs."""
    out = Path(output)
    changes = []
    for split in ['train', 'test']:
        manifest_path = out/split/'manifest.json'
        scoring_path = out/split/'scoring.json'
        manifest = json.loads(manifest_path.read_text())
        scoring = json.loads(scoring_path.read_text())
        by_id = {row['sample_id']: row for row in scoring}
        def schedule(row):
            return decoded_ocp_schedule(row['media']['video']['path'], row['source_id'])
        with ThreadPoolExecutor(workers) as pool:
            schedules = list(pool.map(schedule, manifest['samples']))
        for row, indices in zip(manifest['samples'], schedules):
            score = by_id[row['sample_id']]
            old = score['feature_frame_indices']
            if old != indices:
                changes.append(dict(split=split, sample_id=row['sample_id'],
                    source_id=row['source_id'], previous=old, corrected=indices))
            row['input_metadata']['frame_indices'] = list(dict.fromkeys(indices))
            score['feature_frame_indices'] = indices
        write_json(manifest_path, manifest)
        write_json(scoring_path, scoring)
    write_json(out/'frame_schedule_recovery.json', dict(
        protocol='official_physion_ocp_clip_and_left_pad', changes=changes))
    return changes


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
    with ThreadPoolExecutor(8) as workers:
        schedules=dict(zip(todo,workers.map(
            lambda item: decoded_ocp_schedule(paths[item], item[1]), todo)))
    for split,names in selections.items():
        samples=[];scoring=[]
        for i,name in enumerate(names):
            feature_indices=schedules[split,name]
            indices=list(dict.fromkeys(feature_indices))
            sample_id=f'physion_{split}:{i}'
            samples.append(dict(sample_id=sample_id,source_id=name,question='Physical scene observations.',
                media={'video':{'path':paths[split,name]}},input_metadata={'frame_indices':indices}))
            scoring.append(dict(sample_id=sample_id,name=name,scenario=scenario(name),
                label=labels[split][name]['label'],feature_frame_indices=feature_indices))
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
