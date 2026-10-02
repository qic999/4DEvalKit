"""Compute official joint Omni3D AP after all six subsets have finished."""
import argparse
from pathlib import Path
import time
import numpy as np

from core.io import read_json, write_json
from scripts.wilddet3d_protocol import load_official, UPSTREAM_COMMIT
from scripts.score_wilddet3d_perception import finite_metrics

SUBSETS = ['KITTI', 'nuScenes', 'SUNRGBD', 'Hypersim', 'ARKitScenes', 'Objectron']


def score(root, model, upstream=None, data_root=None):
    load_official(upstream or root/'upstream')
    from wilddet3d.eval.omni3d import Omni3DEvaluator
    evaluator = Omni3DEvaluator(data_root=str(data_root or root/'data/omni3d'), omni3d50=True,
                               datasets=[s+'_test' for s in SUBSETS], per_class_eval=True)
    images = 0
    for subset in SUBSETS:
        manifest = read_json(root/f'omni3d_{subset}_manifest.json')
        ev = evaluator.evaluators[subset+'_test']
        classes = {ev.cat_map[name]:idx for idx,name in ev.id2name.items()}
        for sample in manifest['samples']:
            r = read_json(root/f'omni3d_{subset}'/model/'samples'/f'{sample["image_id"]}.json')
            images += 1
            preds = r['predictions']
            if not preds:
                continue
            ev._predictions_to_coco(r['image_id'], np.array([p['bbox_xyxy'] for p in preds],dtype=np.float32),
                np.array([p['box3d_wlh_wxyz'] for p in preds],dtype=np.float32),
                np.array([p['score'] for p in preds],dtype=np.float32),
                np.array([classes[p['category_id']] for p in preds],dtype=np.int64))
    scores, log = evaluator.evaluate('3D')
    out = root/'omni3d_overall'/model
    write_json(out/'metrics.json', {'benchmark':'omni3d_overall', 'images':images,
        'upstream_commit':UPSTREAM_COMMIT,'aggregation':'Official Omni3DEvaluator joint category/image accumulation',
        'metrics':{'bbox':finite_metrics(scores)}})
    (out/'metrics.txt').write_text(log)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--upstream', help='Official evaluator checkout; defaults to <run-root>/upstream')
    p.add_argument('--data-root', help='Omni3D data directory; defaults to <run-root>/data/omni3d')
    p.add_argument('--models', nargs='+', default=['small_e100', 'full_e64'])
    p.add_argument('--wait', action='store_true')
    args = p.parse_args()
    root = Path(args.run_root).resolve()
    deadline = time.time()+72*3600
    pending = set(args.models)
    while pending:
        for model in sorted(pending):
            if (root/'omni3d_overall'/model/'metrics.json').exists():
                pending.remove(model)
                continue
            statuses = [root/f'omni3d_{s}'/model/'scoring_status.json' for s in SUBSETS]
            if all(p.exists() and read_json(p)['complete'] for p in statuses):
                score(root, model, args.upstream, args.data_root)
                pending.remove(model)
        if not pending:
            break
        if not args.wait or time.time()>deadline:
            raise RuntimeError(f'Six complete Omni3D subsets required for {pending}')
        time.sleep(30)


if __name__ == '__main__':
    main()
