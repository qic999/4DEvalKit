import io
from pathlib import Path
import zipfile

import pytest
from PIL import Image

from core.io import read_json, write_json
from core.media import export_media
from scripts.prepare_benchmarks import extract_archive, prepare_job, resolve_media_path


def test_local_blink_snapshot_only_needs_selected_splits(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from benchmark.loader import BenchmarkSession
    from scripts.benchmark_registry import get_spec
    for subset in ['Counting','Relative_Depth','Spatial_Relation']:
        path = tmp_path/subset/'val-00000-of-00001.parquet'
        path.parent.mkdir()
        pq.write_table(pa.Table.from_pylist([{'idx':subset,'sub_task':subset}]),path)
    session = BenchmarkSession(get_spec('BLINK'),data=str(tmp_path))
    assert session.selected_count == 3
    assert [r['idx'] for r in session.raw] == ['Counting','Relative_Depth','Spatial_Relation']


def test_media_resolution_is_exact_and_rejects_ambiguity(tmp_path):
    a, b = tmp_path / 'a', tmp_path / 'b'
    for root in (a, b):
        (root / 'videos').mkdir(parents=True)
        (root / 'videos/x.mp4').write_bytes(b'video')
    url = 'https://huggingface.co/datasets/author/data/resolve/main/videos/x.mp4'
    assert resolve_media_path(url, [a], 'author/data') == a / 'videos/x.mp4'
    with pytest.raises(FileNotFoundError, match='found 2'):
        resolve_media_path(url, [a,b], 'author/data')
    with pytest.raises(ValueError, match='Unresolved remote'):
        resolve_media_path(url, [a], 'other/data')
    with pytest.raises(ValueError, match='Unsafe'):
        resolve_media_path('../x.mp4', [a])


def test_zip_export_and_traversal(tmp_path):
    raw = io.BytesIO()
    Image.new('RGB', (5, 7), 'red').save(raw, format='PNG')
    archive = tmp_path / 'images.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('images/a.png', raw.getvalue())
    record = export_media({'type': 'zip_image', 'archive': str(archive), 'member': 'images/a.png'}, tmp_path / 'out')
    with Image.open(record['path']) as im:
        assert im.size == (5,7) and im.getpixel((0,0)) == (255,0,0)
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape', b'bad')
    with pytest.raises(ValueError, match='escapes'):
        extract_archive(archive, tmp_path / 'extract')
    assert not (tmp_path / 'escape').exists()


def test_full_preparation_resumes_without_exposing_answers(tmp_path):
    image = tmp_path / 'input.png'
    Image.new('RGB', (8,8)).save(image)
    source = tmp_path / 'qa.json'
    write_json(source, [{'id': 'q1', 'question': 'Which direction?', 'video': str(image),
                         'choices': {'A': 'left', 'B': 'right'}, 'answer': 'left',
                         'private_notes': 'do not export this'}])
    job = {'name': 'toy', 'benchmark': 'VLM4D', 'data': str(source)}
    out = tmp_path / 'prepared'
    assert prepare_job(job, out)['phase'] == 'encoder_inputs_ready'
    manifest = out / 'toy/real_mc/manifest.json'
    original = read_json(manifest)
    assert original['num_samples'] == 1
    assert 'answer' not in original['samples'][0]
    assert 'private_notes' not in manifest.read_text()
    assert prepare_job(job, out)['phase'] == 'encoder_inputs_ready'
    assert read_json(manifest) == original
    source.write_text(source.read_text() + '\n')
    with pytest.raises(ValueError, match='source changed'):
        prepare_job(job, out)
