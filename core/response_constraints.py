"""Bound final-answer syntax using public questions, never answer annotations."""
import re

VERSION = 'native_answer_v1'
# Bounded decimal syntax prevents an indefinitely growing number. These limits
# are intentionally much wider than scene-scale distances; units stay explicit.
NUMBER = r'(0|[1-9][0-9]{0,11})(\.[0-9]{1,8})?'
DISTANCE = r'\\scalar\{' + NUMBER + r'\} \\distance_unit\{(m|cm|mm|ft|in)\}'


def response_constraint(benchmark, question, public_choices=None):
    if benchmark == 'Q-Spatial-Bench':
        return {'regex': DISTANCE}
    if benchmark == 'SAT':
        if not isinstance(public_choices, list) or not public_choices:
            raise ValueError('SAT needs its public answer choices')
        if any(not isinstance(c, str) or not c or c not in question for c in public_choices):
            raise ValueError('SAT choices must occur in the public question')
        return {'choice': sorted(set(public_choices), key=question.index)}
    matches = re.findall(r'(?:^|\n)\s*(?:\(([A-Z])\)|([A-Z])[.):])\s+', question)
    letters = sorted({a or b for a,b in matches})
    if not letters:
        letters = sorted(set(re.findall(r'\(([A-Z])\)\s*', question)))
    if not letters:
        letters = sorted(set(re.findall(r'(?<![A-Za-z0-9])([A-Z])[.:)]\s+', question)))
    if letters:
        if len(letters) < 2 or letters != [chr(65+i) for i in range(len(letters))]:
            raise ValueError('Cannot infer a contiguous public option list')
        return {'choice': letters}
    if benchmark == 'VSI-Bench':
        return {'regex': NUMBER}
    raise ValueError(f'No supported public answer format for {benchmark}')


def matches_constraint(text, constraint):
    text = text.strip()
    if 'choice' in constraint:
        return text in constraint['choice']
    return re.fullmatch(constraint['regex'], text) is not None
