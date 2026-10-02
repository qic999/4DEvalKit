"""Score complete prediction coverage with the unmodified official evaluator."""
import argparse
from pathlib import Path
import numpy as np

from core.io import read_json, write_json, source_signature
from scripts.wilddet3d_protocol import load_official, evaluator_kwargs, UPSTREAM_COMMIT


def finite_metrics(scores):
    """Keep undefined official metrics as null, never as a fabricated zero."""
    return {k: float(v) if np.isfinite(v) else None for k, v in scores.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['upstream', 'manifest', 'predictions', 'output']:
        p.add_argument('--'+key, required=True)
    p.add_argument('--metrics', nargs='+', choices=['dist', 'bbox'], default=['dist', 'bbox'])
    args = p.parse_args()
    official = load_official(args.upstream)
    m = read_json(args.manifest)
    configs = list(Path(args.predictions).glob('config_*.json'))
    if not configs or any(read_json(c)['manifest_signature'] != source_signature(args.manifest) for c in configs):
        raise ValueError('Prediction manifest identity does not match scoring input')
    annotation = m['annotation']['path']
    if source_signature(annotation) != m['annotation']:
        raise ValueError('Scoring annotations changed after manifest preparation')
    if m.get('evaluation_scope'):
        ground_truth_ids = {im['id'] for im in read_json(annotation)['images']}
        manifest_ids = {im['image_id'] for im in m['samples']}
        if ground_truth_ids != manifest_ids or len(ground_truth_ids) != m['evaluation_scope']['evaluated_images']:
            raise ValueError('Subset ground truth does not match manifest coverage')
    kwargs = evaluator_kwargs(m['benchmark'], annotation)
    id2class = {cid: kwargs['det_map'][name] for name, cid in kwargs['cat_map'].items() if name in kwargs['det_map']}
    rows = []
    for sample in m['samples']:
        path = Path(args.predictions)/'samples'/f'{sample["image_id"]}.json'
        row = read_json(path)  # Missing images must never silently disappear from AP.
        if row['image_id'] != sample['image_id']:
            raise ValueError('Image ID mismatch')
        rows.append(row)
    result = {'benchmark': m['benchmark'], 'images': len(rows),
              'predictions': sum(len(x['predictions']) for x in rows),
              'upstream_commit': UPSTREAM_COMMIT, 'protocol': m['protocol'],
              'annotation': m['annotation'], 'evaluation_scope': m.get('evaluation_scope', {'type': 'full_split'}),
              'metric_units': 'Official raw values; AP/ODS multiplied by 100 in summary tables', 'metrics': {}}
    for metric in args.metrics:
        evaluator = official.Detect3DEvaluator(iou_type=metric, **kwargs)
        for row in rows:
            predictions = row['predictions']
            if not predictions:
                continue
            evaluator._predictions_to_coco(row['image_id'],
                np.array([x['bbox_xyxy'] for x in predictions], dtype=np.float32),
                np.array([x['box3d_wlh_wxyz'] for x in predictions], dtype=np.float32),
                np.array([x['score'] for x in predictions], dtype=np.float32),
                np.array([id2class[x['category_id']] for x in predictions], dtype=np.int64))
        scores, log = evaluator.evaluate('3D')
        result['metrics'][metric] = finite_metrics(scores)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(f'.{metric}.txt').write_text(log)
        write_json(out, result)
        print(metric, result['metrics'][metric], flush=True)


if __name__ == '__main__':
    main()
