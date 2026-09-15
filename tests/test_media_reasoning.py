import io
import json
import sys

from core.io import read_json, write_json
from scripts import run_media_models as pipeline


def test_qa_only_uses_all_cached_geometry_and_explicit_budget(tmp_path, monkeypatch):
    source, output = tmp_path/'original', tmp_path/'long_answers'
    manifest = tmp_path/'manifest.json'
    write_json(manifest, {'samples':[{'sample_id':'a'}, {'sample_id':'b'}]})
    scene = {'units':'m', 'coordinate_frame':'camera', 'objects':[]}
    variants = [{'name':'full'}, {'name':'small'}]
    for variant in variants:
        write_json(source/'job'/variant['name']/'geometry.json', {'scenes':{'a':scene, 'b':scene}})
        write_json(source/'job'/variant['name']/'qa.json', {'original':True})
    config = {'jobs':[{'name':'job', 'manifest':str(manifest), 'benchmark':'SAT',
                      'data':'unused', 'split':'test'}], 'variants':variants,
              'llm_python':'unused-python', 'llm_model':'unused-model', 'llm_name':'llm',
              'llm_gpu':7, 'llm_port':23412, 'qa_max_tokens':4096, 'qa_timeout':300}
    config_path = tmp_path/'config.json'
    write_json(config_path, config)
    calls = []
    class FakeProcess:
        def __init__(self, command, **kwargs):
            calls.append(command)
            self.pid = 90000 + len(calls)
            self.server = 'vllm.entrypoints.openai.api_server' in command
            if not self.server:
                assert command[1] == 'eval.py'
                target = command[command.index('--output')+1]
                write_json(target, {'num_samples':2, 'status_counts':{'ok':2},
                                    'primary_metric':{'score_100':50}})
        def poll(self):
            return None if self.server else 0
        def wait(self):
            return 0
    monkeypatch.setattr(pipeline.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(pipeline.subprocess, 'check_output', lambda *a,**kw:'7, 0\n')
    monkeypatch.setattr(pipeline.urllib.request, 'urlopen',
                        lambda *a,**kw:io.StringIO(json.dumps({'data':[{'id':'llm'}]})))
    monkeypatch.setattr(pipeline.os, 'killpg', lambda *a:None)
    monkeypatch.setattr(pipeline.signal, 'signal', lambda *a:None)
    monkeypatch.setattr(sys, 'argv', ['run_media_models', '--config',str(config_path),
                                    '--geometry-root',str(source), '--output-root',str(output)])
    pipeline.main()
    assert len(calls) == 3  # One reasoner and two QA jobs; no detector or encoders.
    for command,variant in zip(calls[1:], variants):
        assert command[command.index('--max-tokens')+1] == '4096'
        assert command[command.index('--timeout')+1] == '300'
        assert command[command.index('--geometry')+1] == str(source/'job'/variant['name']/'geometry.json')
        assert '--limit' not in command
        assert read_json(source/'job'/variant['name']/'qa.json') == {'original':True}
    status = read_json(output/'status.json')['jobs']['job']
    assert status['phase'] == 'complete'
    assert all(c['samples'] == 2 for c in status['coverage'].values())
