"""Audit exactly the MVVBench views needed by the selected questions.

Optionally link locally authorized source videos and prepare a full or explicitly
named source subset. Never silently skip questions with missing camera views.
"""
import argparse
from collections import Counter
import csv
from pathlib import Path, PurePosixPath

from core.io import read_json, source_signature, write_json


def relative_path(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or '..' in p.parts:
        raise ValueError(f'Unsafe relative video path: {value!r}')
    return p


def source_path(row, root):
    path = relative_path(row['source_rel_path'])
    if row['dataset'] == 'egoexo':
        return root/'egoexo'/'takes'/path
    if row['dataset'] == 'mmptrack':
        return root/'mmptrack'/relative_path(row['split'] or 'train')/'videos'/path
    if row['dataset'] == 'panoptic':
        return root/'panoptic'/path
    raise ValueError(f"Unsupported source: {row['dataset']}")


def audit(annotations, mapping, videos, output, *, source='all', source_root=None, link=False):
    questions = read_json(annotations)
    with Path(mapping).open() as stream:
        rows = list(csv.DictReader(stream))
    by_target = {r['target']: r for r in rows}
    if len(by_target) != len(rows):
        raise ValueError('Duplicate target in video mapping')
    scene_sources = {}
    for row in rows:
        relative_path(row['target'])
        if '/' in row['target'] or row['target'] != f"{row['sample_name']}_view{row['view']}.mp4":
            raise ValueError('Video target disagrees with sample and view IDs')
        old = scene_sources.setdefault(row['sample_name'], row['dataset'])
        if old != row['dataset']:
            raise ValueError('One sample maps to multiple source datasets')
    if set(questions) - set(scene_sources):
        raise ValueError('Annotations contain unmapped scenes')
    selected = {k: qs for k, qs in questions.items() if source == 'all' or scene_sources[k] == source}
    if not selected:
        raise ValueError('No selected questions')
    targets = sorted({f'{k}_view{v}.mp4' for k, qs in selected.items() for q in qs for v in q['views']})
    unknown = set(targets) - set(by_target)
    videos, out = Path(videos).resolve(), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    if link and not source_root:
        raise ValueError('--link requires --source-root')
    missing, linked, available = [], [], set()
    for target in targets:
        if target in unknown:
            # Preserve broken upstream references in the inventory. Never invent
            # a replacement camera or remove the associated question.
            missing.append(dict.fromkeys(list(rows[0])+['local_source'], ''))
            missing[-1].update(target=target, dataset='unmapped')
            continue
        row = by_target[target]
        dest = videos/target
        original = source_path(row, Path(source_root).resolve()) if source_root else None
        if link and not dest.exists() and original is not None and original.is_file():
            if dest.is_symlink():
                raise ValueError(f'Refusing to replace broken link: {dest}')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(original)
            linked.append(target)
        if dest.is_file() and dest.stat().st_size > 0:
            available.add(target)
        else:
            missing.append(dict(row, local_source=str(original) if original else ''))
    ready_questions = sum(all(f'{k}_view{v}.mp4' in available for v in q['views'])
                          for k, qs in selected.items() for q in qs)
    report = dict(source=source, phase='media_present' if not missing else 'missing_videos',
        total_questions=sum(len(qs) for qs in selected.values()), complete_questions=ready_questions,
        required_videos=len(targets), available_videos=len(available), missing_videos=len(missing),
        missing_by_source=dict(Counter(r['dataset'] for r in missing)), linked=linked,
        unmapped_views=sorted(unknown),
        annotations=source_signature(annotations), mapping=source_signature(mapping),
        selection='all_questions' if source == 'all' else f'all_questions_from_{source}',
        note='File-presence audit; preparation additionally validates video decoding.')
    write_json(out/'inventory.json', report)
    write_json(out/'selected_questions.json', selected)
    with (out/'missing_videos.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])+['local_source'])
        writer.writeheader(); writer.writerows(missing)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations', required=True); p.add_argument('--mapping', required=True)
    p.add_argument('--videos', required=True); p.add_argument('--output', required=True)
    p.add_argument('--source', choices=['all', 'egoexo', 'mmptrack', 'panoptic'], default='all')
    p.add_argument('--source-root'); p.add_argument('--link', action='store_true')
    p.add_argument('--prepare', action='store_true')
    a = p.parse_args()
    report = audit(a.annotations, a.mapping, a.videos, a.output, source=a.source,
                   source_root=a.source_root, link=a.link)
    if a.prepare:
        if report['missing_videos']:
            raise FileNotFoundError(f"{report['missing_videos']} required videos missing; see missing_videos.csv")
        from scripts.prepare_additional_4d import prepare
        prepare('MVVBench', Path(a.output)/'selected_questions.json', [a.videos], Path(a.output)/'prepared')
    print(__import__('json').dumps(report, indent=2))


if __name__ == '__main__':
    main()
