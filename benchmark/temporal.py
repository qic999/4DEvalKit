"""Additional temporal QA adapters. Annotations never enter media manifests.

MotionBench uses labeled DEV only; CLEVRER retains per-option/per-question
metrics; TempCompass uses the upstream disable_llm matching policy. MVVBench
uses local exact-choice scoring, not an unverified official scorer.
"""
from collections import defaultdict
import json
from pathlib import Path
import re

from benchmark.dynamic import DynamicDataset, load_rows
from core.answers import final_text


def embedded_choices(question):
    hits = list(re.finditer(r'(?:^|\n)\s*([A-Z])[.)]\s+', question))
    if not hits:
        raise ValueError('Missing public multiple-choice options')
    return {m[1]: question[m.end():hits[i+1].start() if i+1 < len(hits) else len(question)].strip()
            for i, m in enumerate(hits)}


def tempcompass_match(output, answer):
    if output is None:
        return False
    output = output.replace('</s>', '').strip()
    if output == answer:
        return True
    if output in 'ABCD' and len(output) == 1:
        return output == answer[0]
    if any(output.startswith(x + sep) for x in 'ABCD' for sep in '.)'):
        return output[0] == answer[0]
    return False


def fourd_option(output):
    if not isinstance(output, str):
        return None
    hits = re.findall(r'\(([A-D])\)', output)
    if not hits:
        hits = re.findall(r'(?:^|[\s\(\.,;:])([A-D])(?:[\s\)\.,;:]|$)', output)
    return hits[0] if hits else None


class TemporalDataset(DynamicDataset):
    def __init__(self, task_name, dataset_name, split=None, **kwargs):
        super().__init__(task_name, dataset_name, split, **kwargs)
        self.protocol = {
            'MotionBench': 'motionbench_labeled_dev_exact_v1',
            'TempCompass': 'tempcompass_mc_disable_llm_v1',
            'CLEVRER': 'clevrer_validation_option_and_question_v1',
            '4D-Bench': '4dbench_qa_views_1_8_16_v1',
            'MVVBench': 'mvvbench_local_exact_choice_v1',
        }[task_name]

    def load_dataset(self):
        p = Path(self.dataset_name)
        if not p.is_file():
            raise ValueError(f'{self.task_name} needs a local annotation file; see scripts.prepare_additional_4d')
        if p.suffix == '.jsonl':
            raw = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        else:
            raw = load_rows(p) if p.suffix == '.parquet' else json.loads(p.read_text())
        # Preparation preserves these normalized records and their stable IDs.
        if isinstance(raw, dict) and raw.get('schema_version') == '4deval.temporal.v1':
            if raw['benchmark'] != self.task_name:
                raise ValueError('Prepared annotation benchmark mismatch')
            self.source_available_count = raw.get('source', {}).get('available', len(raw['samples']))
            return raw['samples']
        rows = []
        if self.task_name == 'MotionBench':
            if self.split != 'dev':
                raise ValueError('Only labeled MotionBench DEV is locally scorable')
            for v in raw:
                for q in v['qa']:
                    if q['answer'] == 'NA':
                        continue  # Published hidden TEST; never impute its labels.
                    rows.append(dict(id=q['uid'], question=q['question'], answer=q['answer'],
                                     choices=embedded_choices(q['question']), video=v['video_path'],
                                     question_type=v['question_type'], scene=v.get('video_type', 'unknown')))
        elif self.task_name == 'TempCompass':
            for i, r in enumerate(raw):
                rows.append(dict(id=f"{r['video_id']}:{i}", question=r['question'],
                                 answer=r['answer'][0], native_answer=r['answer'],
                                 choices=embedded_choices(r['question']), video=r['video_id']+'.mp4',
                                 question_type=r['dim'], scene=r['video_id']))
        elif self.task_name == 'MVVBench':
            for scene, questions in raw.items():
                for i, q in enumerate(questions):
                    rows.append(dict(id=f'{scene}:{i}', question=q['question'], choices=q['choices'],
                                     answer=q['correct_answer'], question_type=q['category'],
                                     scene=scene, static_or_dynamic=q['static_or_dynamic'],
                                     view_ids=[str(v) for v in q['views']],
                                     video=[f'{scene}_view{v}.mp4' for v in q['views']]))
                    # Relevant-interval labels are NOT used to crop model input.
        elif self.task_name == 'CLEVRER':
            if self.split != 'validation':
                raise ValueError('CLEVRER local scoring requires validation labels')
            for v in raw:
                for q in v['questions']:
                    group = f"{v['scene_index']}:{q['question_id']}"
                    base = dict(question=q['question'], video=v['video_filename'], scene=str(v['scene_index']),
                                question_type=q['question_type'], group_id=group)
                    if q['question_type'] == 'descriptive':
                        rows.append(dict(base, id=group, answer=str(q['answer']), choices=None,
                                         group_size=1))
                    else:
                        for c in q['choices']:
                            rows.append(dict(base, id=f"{group}:{c['choice_id']}",
                                question=q['question']+'\nCandidate: '+c['choice']+'\nIs this candidate correct?',
                                choices={'A': 'correct', 'B': 'wrong'},
                                answer={'correct': 'A', 'wrong': 'B'}[c['answer']], group_size=len(q['choices'])))
        elif self.task_name == '4D-Bench':
            for uid, q in raw.items():
                question = q.get('Question', q.get('question'))
                choices = {k: q[k] for k in 'ABCD'} if 'A' in q else q.get('Options', q.get('choices'))
                if choices is None:
                    # Released option keys can be "A. ..."; never stringify the
                    # entire annotation because that includes Answer index.
                    choices = {k.strip('().'): value for k,value in q.items() if re.fullmatch(r'\(?[A-D][.)]?', k)}
                if not choices:
                    choices = embedded_choices(question)
                if isinstance(choices, list):
                    choices = dict(zip('ABCD', choices))
                rows.append(dict(id=uid, question=question, choices=choices,
                    answer=chr(64+int(q['Answer index'])), question_type=q.get('Category', 'unknown'),
                    scene=uid.split('_')[0], view_ids=['1', '8', '16'],
                    video=[f"{uid.split('_')[0]}/view_{v}_rgb_white_bg.mp4" for v in [1,8,16]]))
        if not rows:
            raise ValueError('No labeled evaluation samples')
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate native QA IDs')
        return rows

    def prepare_dataset(self, rows):
        samples = []
        for r in rows:
            question = r['question'].strip()
            options = r.get('choices')
            if options:
                if not re.search(r'(?:^|\n)\s*A[.)]\s+', question):
                    question += '\n' + '\n'.join(f'{k}. {v}' for k,v in options.items())
                question += '\nAnswer with only the option letter.'
            else:
                question += '\nAnswer with only a single word or integer.'
            meta = {k: r[k] for k in ['id','question_type','scene','choices','native_answer',
                    'group_id','group_size','static_or_dynamic','view_ids'] if k in r}
            meta['video_path'] = r['video']
            samples.append(dict(question=question, answer=r['answer'], video=r['video'], metadata=meta))
        return samples

    def evaluate_results(self, samples, outputs):
        if len(samples) != len(outputs):
            raise ValueError('Every sample requires a prediction (invalid answers score zero)')
        results = []
        for sample, output in zip(samples, outputs):
            meta, gt = sample['metadata'], sample['answer']
            text = final_text(output or '').strip()
            if self.task_name == 'TempCompass':
                correct = tempcompass_match(output, meta['native_answer'])
            elif self.task_name == '4D-Bench':
                correct = fourd_option(output) == gt
            else:
                # MotionBench's official interface is an exact letter mapping.
                # CLEVRER descriptive strings keep the native answer vocabulary.
                correct = text == str(gt)
            results.append(dict(metadata=meta, ground_truth=gt, processed_answer=text, is_correct=correct))
        return results

    def compute_statistics(self, results):
        stats = super().compute_statistics(results)
        if self.task_name == 'CLEVRER':
            groups = defaultdict(list)
            for r in results:
                groups[r['metadata']['group_id']].append(r)
            by_type = defaultdict(list)
            for group in groups.values():
                if len(group) != group[0]['metadata']['group_size']:
                    raise ValueError('Incomplete CLEVRER question options; cannot score per-question accuracy')
                by_type[group[0]['metadata']['question_type']].append(all(r['is_correct'] for r in group))
            stats['per_question'] = {k: {'accuracy': sum(v)/len(v), 'count': len(v)} for k,v in by_type.items()}
            stats['overall_question_accuracy'] = sum(sum(v) for v in by_type.values())/len(groups)
        if self.task_name == 'MVVBench':
            kinds = defaultdict(list)
            for r in results:
                kinds[r['metadata']['static_or_dynamic']].append(r['is_correct'])
            stats['per_scene_dynamics'] = {k: {'accuracy': sum(v)/len(v), 'count': len(v)} for k,v in kinds.items()}
        return stats
