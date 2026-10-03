"""Normalize new temporal QA annotations and validate every selected video.

Example: python -m scripts.prepare_additional_4d --benchmark TempCompass
  --annotations data/TempCompass/multi-choice/test-00000-of-00001.parquet
  --video-root data/TempCompass/videos --output data/prepared/TempCompass
Use --limit for an explicitly recorded smoke subset, omit it for the full split.
"""
import argparse
from collections import Counter
from pathlib import Path

from benchmark.loader import BenchmarkSession, media_manifest
from core.io import digest, source_signature, write_json
from scripts.benchmark_registry import get_spec
from scripts.prepare_benchmarks import validate_media


def prepare(benchmark, annotations, video_roots, output, limit=None):
    spec = get_spec(benchmark)
    session = BenchmarkSession(spec, data=str(annotations), limit=limit)
    out = Path(output).resolve(); out.mkdir(parents=True, exist_ok=True)
    rows = []; manifest = []; cache = {}
    for batch in session.batches():
        for example in batch:
            entry = media_manifest(example)
            entry['media']['video'] = validate_media(entry['media']['video'], video_roots, cache=cache)
            paths = entry['media']['video']
            row = dict(session.raw[example['row_index']])
            row['video'] = [p['path'] for p in paths] if isinstance(paths, list) else paths['path']
            # Re-index the frozen subset: data and manifest IDs must agree.
            entry['sample_id'] = f'{spec.slug}:{len(rows)}'
            entry['input_metadata']['video_path'] = row['video']
            rows.append(row); manifest.append(entry)
    source = {'annotations': source_signature(annotations), 'available': session.available_count,
              'selected': len(rows), 'limit': limit, 'selection': 'source_order_with_complete_question_groups',
              'split': session.split, 'protocol': session.protocol}
    write_json(out/'annotations.json', dict(schema_version='4deval.temporal.v1', benchmark=benchmark,
               source=source, samples=rows))
    write_json(out/'manifest.json', dict(benchmark=benchmark, split=session.split, num_samples=len(rows),
               source=source, samples=manifest))
    write_json(out/'status.json', dict(phase='encoder_inputs_ready', samples=len(rows), videos=len(cache),
               categories=dict(Counter(r['question_type'] for r in rows)), manifest_digest=digest(manifest), **source))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--video-root', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--limit', type=int)
    args = p.parse_args()
    print(prepare(args.benchmark, args.annotations, args.video_root, args.output, args.limit))


if __name__ == '__main__':
    main()
