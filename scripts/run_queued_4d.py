"""Start a background stage only after explicit dependency statuses complete."""
import argparse
import os
from pathlib import Path
import subprocess
import time

from core.io import read_json,write_json
from core.runner import output_lock


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--status',required=True);a=p.parse_args();c=read_json(a.config)
    state=dict(pid=os.getpid(),phase='waiting',dependencies=c['dependencies'])
    with output_lock(a.status):
        try:
            while True:
                pending=[]
                for d in c['dependencies']:
                    path=Path(d['path']);actual=read_json(path).get('phase') if path.is_file() else 'missing'
                    if actual in {'failed','interrupted'}:raise RuntimeError(f'Dependency {path}: {actual}')
                    if actual!=d['phase']:pending.append(dict(path=str(path),phase=actual))
                state.update(pending=pending,updated=time.time());write_json(a.status,state)
                if not pending:break
                time.sleep(15)
            state.update(phase='running',command=c['command']);write_json(a.status,state)
            subprocess.run(c['command'],check=True,cwd=c.get('cwd'))
            state.update(phase='complete',updated=time.time());write_json(a.status,state)
        except Exception as exc:
            state.update(phase='failed',error=str(exc),updated=time.time());write_json(a.status,state);raise


if __name__=='__main__':main()
