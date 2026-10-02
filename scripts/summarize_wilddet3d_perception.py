"""Collect raw perception metrics and keep paper references explicitly labeled."""
import argparse
import csv
import io
import os
from pathlib import Path
import tempfile
import time

from core.io import read_json


DATASETS = ['stereo4d', 'scannet', 'argoverse', 'omni3d_KITTI', 'omni3d_nuScenes',
            'omni3d_SUNRGBD', 'omni3d_Hypersim', 'omni3d_ARKitScenes', 'omni3d_Objectron', 'omni3d_overall', 'in_the_wild']
PAPER = {'stereo4d': ('AP (distance)', 7.5, 'Box prompt, Table 6'),
         'scannet': ('ODS (canonical)', 48.9, 'Canonical rotation per section 4.1; Table 5 does not explicitly label prompt mode'),
         'argoverse': ('ODS (canonical)', 40.3, 'Canonical rotation per section 4.1; Table 5 does not explicitly label prompt mode'),
         'omni3d_KITTI': ('AP (IoU)', 44.3, 'Box prompt, Table 4'),
         'omni3d_nuScenes': ('AP (IoU)', 35.3, 'Box prompt, Table 4'),
         'omni3d_SUNRGBD': ('AP (IoU)', 43.1, 'Box prompt, Table 4'),
         'omni3d_Hypersim': ('AP (IoU)', 17.3, 'Box prompt, Table 4'),
         'omni3d_ARKitScenes': ('AP (IoU)', 66.6, 'Box prompt, Table 4'),
         'omni3d_Objectron': ('AP (IoU)', 60.8, 'Box prompt, Table 4'),
         'omni3d_overall': ('AP (IoU)', 36.4, 'Box prompt, joint Omni3D AP, Table 4'),
         'in_the_wild': ('AP (distance)', 24.8, 'Box prompt, full-data model, Table 3')}


def summarize(root):
    root = Path(root)
    config_path = root/'comparison_config.json'
    config = read_json(config_path) if config_path.exists() else {}
    models = config.get('models', [{'name': 'full_e64', 'label': 'Full e64'},
                                   {'name': 'small_e100', 'label': 'Small WDS518 e100'}])
    statuses = [read_json(root/p) for p in config.get('status_files', ['status.json']) if (root/p).exists()]
    jobs_all = [j for status in statuses for j in status.get('jobs', [])]
    scoring = {key for status in statuses for key in status.get('scoring', [])}
    rows = []
    subset_notes = []
    lines = ['# Perception evaluation', '',
        'Inputs: single RGB image and GT 2D box prompts. Predictions: metric oriented 3D boxes. No LLM, GT depth, GT camera intrinsics or GT 3D boxes are supplied to the encoder.', '',
        'Scoring uses the frozen WildDet3D evaluator (`1b8aa52b6ff3f00d0ebfa07175efc0c0c440964a`). Scores below are on a 0–100 scale. Missing results are not zero scores.', '',
        '| Dataset | Metric | ' + ' | '.join(model['label'] for model in models) + ' | WildDet3D paper, no GT depth | State |',
        '|---|---|' + '---:|'*len(models) + '---:|---|']
    for dataset in DATASETS:
        manifest_path = root/f'{dataset}_manifest.json'
        scope = read_json(manifest_path).get('evaluation_scope', {}) if manifest_path.exists() else {}
        label = dataset
        if scope.get('type') == 'explicit_image_subset':
            coverage = f'{scope["evaluated_images"]}/{scope["original_images"]}'
            label += f' (subset: {coverage} images)'
            subset_notes.append(f'- **{dataset}: {coverage} images.** {scope["excluded_images"]} explicitly excluded images and all their annotations are removed from both inference and scoring. The paper value uses the full split and is not a matched-subset comparison. Category frequency groups are recomputed on this subset by the official evaluator; exclusions are recorded in `{manifest_path.name}`.')
        metric_name, paper, reference = PAPER[dataset]
        mode = 'bbox' if dataset.startswith('omni3d_') else 'dist'
        key = 'ODS_Canonical' if dataset in ['scannet', 'argoverse'] else 'AP'
        values, states = [], []
        for model_info in models:
            model = model_info['name']
            path = root/dataset/model/'metrics.json'
            metrics = read_json(path) if path.exists() else {}
            scores = metrics.get('metrics', {}).get(mode)
            value = scores[key]*100 if scores is not None and scores.get(key) is not None else None
            values.append(f'{value:.2f}' if value is not None else '—')
            jobs = [j for j in jobs_all if j['dataset']==dataset and j['model']==model]
            scoring_path = root/dataset/model/'scoring_status.json'
            scoring_failed = scoring_path.exists() and not read_json(scoring_path).get('complete')
            state = ('scored' if value is not None else 'failed' if scoring_failed or any(j['state']=='failed' for j in jobs)
                     else 'scoring' if dataset+'/'+model in scoring
                     else 'running' if any(j['state']=='running' for j in jobs)
                     else 'queued' if jobs else 'preparing data')
            if dataset == 'in_the_wild' and value is None and (root/'in_the_wild_download_status.json').exists():
                download = read_json(root/'in_the_wild_download_status.json')
                if download['state'] == 'failed':
                    state = 'data preparation failed (see download status)'
            states.append(f'{model}: {state}')
            rows.append({'dataset': dataset, 'model': model, 'metric': metric_name, 'score_100': value,
                         'images': metrics.get('images'), 'paper_score_100': paper, 'paper_protocol': reference,
                         'evaluation_scope': scope.get('type', 'full_split'),
                         'evaluated_images': scope.get('evaluated_images', metrics.get('images')),
                         'original_images': scope.get('original_images', metrics.get('images')),
                         'excluded_images': scope.get('excluded_images', 0),
                         'state': state, 'source': str(path) if metrics else ''})
        lines.append(f'| {label} | {metric_name} | ' + ' | '.join(values) + f' | {paper:.1f} | {"; ".join(states)} |')
    lines += ['', '## Reading the comparison', '', *subset_notes,
        *['- '+note for note in config.get('notes', [])],
        '- These are box-conditioned 3D regression scores, with GT category IDs attached for scoring; they do not measure open-vocabulary 2D category discovery.',
        '- Table 5 in the paper does not explicitly label its prompt mode. Its ScanNet and Argoverse2 values are references, not a confirmed matched-prompt comparison.',
        '- Model training distributions differ. The Small WDS518 e100 training configuration includes Omni3D and WildDet3D; do not describe all rows as zero-shot. Official evaluation splits are retained; cross-source training-image overlap has not been fully audited.',
        '- Stereo4D uses 383 independent images from videos. It measures perception in dynamic scenes, not temporal tracking or trajectory consistency.',
        '- ODS_Canonical is used above because paper section 4.1 specifies canonical rotations for mAOE. The evaluator also retains raw ODS/ODS_Sym in metrics.json. ATE here is normalized, not a translation error in meters.',
        '- The old ScanNet val88 multi-frame AABB results use a different protocol and are excluded from this table.', '',
        'Paper: https://arxiv.org/html/2604.08626v1 (Tables 3–6). Evaluator: https://github.com/allenai/WildDet3D.', '']
    csv_text = io.StringIO(newline='')
    writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    for name, content in [('comparison.md', '\n'.join(lines)), ('comparison.csv', csv_text.getvalue())]:
        with tempfile.NamedTemporaryFile(mode='w', dir=root, prefix=name+'.', delete=False) as f:
            f.write(content)
        os.replace(f.name, root/name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--watch', action='store_true')
    args = p.parse_args()
    deadline = time.time()+72*3600
    while True:
        summarize(args.run_root)
        if not args.watch or time.time() > deadline:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
