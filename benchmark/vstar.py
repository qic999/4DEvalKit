"""V-STaR's five conditioned subtasks, native outputs and joint metrics.

The released inference example supplies GT intervals to spatial chain 1,
GT box counts to spatial chain 2, and untimed GT boxes to temporal chain 2.
These are declared task conditions, not end-to-end model predictions.
"""
import ast
import json
import math
import re

PROTOCOL = 'vstar_upstream_conditioned_v1'
TASKS = ('answer_vqa', 'answer_temporal', 'answer_spatial',
         'answer_spatial_2', 'answer_temporal_2')
CONDITIONS = {
    'answer_vqa': [],
    'answer_temporal': ['released_temporal_question'],
    'answer_spatial': ['released_spatial_question', 'gt_temporal_interval'],
    'answer_spatial_2': ['released_spatial_question_2', 'gt_box_count'],
    'answer_temporal_2': ['released_temporal_question', 'gt_boxes_without_timestamps'],
}


def task_prompts(row):
    duration = round(row['frame_count'] / row['fps'], 2)
    size = f"{row['width']} x {row['height']}"
    interval = list(range(math.ceil(row['timestamps'][0]), math.floor(row['timestamps'][1])+1))
    boxes = [[b[k] for k in ('xmin', 'ymin', 'xmax', 'ymax')] for b in row['bboxes']]
    prefix = f'The full video duration is {duration} seconds. '
    temporal = ' Return only a JSON array [start_seconds, end_seconds].'
    spatial = (f' Return only a JSON object mapping whole-second timestamps to [x1,y1,x2,y2] boxes.'
               f' Coordinates must be in original {size} image pixels, even if displayed images are resized.')
    return {
        'answer_vqa': f"Answer the question about the video: {row['question']}\n"
                      'If the answer is a person, you do not need to identify the person.',
        'answer_temporal': prefix + row['temporal_question'] + temporal,
        'answer_spatial': row['spatial_question'] + f' Output a box for each second in {interval}.' + spatial,
        'answer_spatial_2': row['spatial_question_2'] +
                           f' Output {len(boxes)} boxes; determine their timestamps yourself.' + spatial,
        'answer_temporal_2': prefix + row['temporal_question'] +
                            f' The key object has these untimed boxes in original {size} pixels: {boxes}.' + temporal,
    }


def parse_prediction(task, text):
    if task == 'answer_vqa':
        return text.strip() or None
    cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        value = json.loads(cleaned)
    except (ValueError, TypeError):
        return None

    def vector(v, n):
        return (isinstance(v, list) and len(v) == n and
                all(type(x) in (int, float) and math.isfinite(x) for x in v))

    if task.startswith('answer_temporal'):
        return value if vector(value, 2) and value[1] >= value[0] >= 0 else None
    if not isinstance(value, dict):
        return None
    if any(not re.fullmatch(r'0|[1-9][0-9]*', k) or not vector(v, 4) or
           v[2] < v[0] or v[3] < v[1] for k, v in value.items()):
        return None
    return value


def judge_namespace(path, model, tokenizer):
    """Use the pinned judge prompt, template and generation function unchanged."""
    tree = ast.parse(path.read_text())
    ns = dict(model=model, tokenizer=tokenizer)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('system_prompt', 'tmpl'):
                    ns[target.id] = ast.literal_eval(node.value)
    from core.official_4d import functions
    functions(path, ['qwen2_5_evaluation'], ns)
    return ns


def summarize(rows):
    """Upstream formulas; invalid outputs remain in every denominator."""
    if not rows:
        raise ValueError('Cannot score an empty selection')
    mean = lambda xs: sum(xs)/len(xs)
    acc = mean([int(r['rating'] >= 2) for r in rows])
    result = dict(samples=len(rows), vqa_accuracy=acc,
                  invalid_judge_outputs=sum(r['rating'] == -1 for r in rows), chains={})
    for chain in ('1', '2'):
        t = [r['chains'][chain]['temporal_iou'] for r in rows]
        s = [r['chains'][chain]['spatial_iou'] for r in rows]
        values = (acc, mean(t), mean(s))
        # The author's log(0) raises on perfect fixtures. Represent that limit
        # explicitly without serializing nonstandard JSON Infinity.
        lgm = None if max(values) == 1 else -sum(math.log1p(-v) for v in values)/3
        result['chains'][chain] = dict(
            temporal_iou=mean(t), spatial_iou=mean(s),
            temporal_recall={str(k): mean([int(v >= k) for v in t]) for k in (.3, .5, .7)},
            spatial_ap={str(k): mean([r['chains'][chain]['spatial_ap'][i] for r in rows])
                        for i, k in enumerate((.1, .3, .5, .7, .9))},
            AM=mean(values), LGM=lgm, LGM_infinite=lgm is None,
            joint_vqa_temporal=mean([int(r['rating'] >= 2 and a >= .3) for r, a in zip(rows, t)]),
            joint_vqa_spatial=mean([int(r['rating'] >= 2 and b >= .1) for r, b in zip(rows, s)]),
            joint_temporal_spatial=mean([int(a >= .3 and b >= .1) for a, b in zip(t, s)]),
            joint_all=mean([int(r['rating'] >= 2 and a >= .3 and b >= .1)
                            for r, a, b in zip(rows, t, s)]))
    result['mAM'] = mean([c['AM'] for c in result['chains'].values()])
    result['mLGM'] = (None if any(c['LGM_infinite'] for c in result['chains'].values())
                     else mean([c['LGM'] for c in result['chains'].values()]))
    return result
