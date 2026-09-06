#!/usr/bin/env python3
"""Install TurboCider-owned, pinned dependencies; never locates sibling projects/models."""
import argparse,json,os,platform,subprocess,sys,venv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');p.add_argument('--wheelhouse',type=Path);p.add_argument('--app-toolchain',type=Path)
    a=p.parse_args()
    if sys.version_info[:2]!=(3,11):raise SystemExit('Run setup with CPython 3.11; no interpreter or models are downloaded by this tool.')
    if platform.system()!='Darwin' or platform.machine()!='arm64':raise SystemExit('Apple Silicon macOS required')
    root=Path(__file__).resolve().parents[1]
    app=a.app_toolchain or Path.home()/'Library/Application Support/TurboCiderNative/toolchains/coreml'
    environment=os.environ.copy();environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None);environment['PYTHONNOUSERSITE']='1'
    for destination,lock in [(root/'.venv',root/'tools/dependencies/build.lock.txt'),(app,root/'tools/dependencies/coreml.lock.txt')]:
        destination=destination.absolute()
        if destination.is_symlink():raise SystemExit('Toolchain must not be a symlink')
        if not (destination/'pyvenv.cfg').exists():
            if destination.exists() and any(destination.iterdir()):raise SystemExit('Refusing to adopt a nonempty non-venv directory')
            venv.EnvBuilder(with_pip=True).create(destination)
        command=[str(destination/'bin/python3'),'-m','pip','install','--requirement',str(lock)]
        if a.offline:command+=['--no-index']
        if a.wheelhouse:command+=['--find-links',str(a.wheelhouse.resolve())]
        subprocess.run(command,env=environment,check=True)
        subprocess.run([str(destination/'bin/python3'),'-m','pip','check'],env=environment,check=True)
        (destination/'turbocider-toolchain.json').write_text(json.dumps({'owner':'TurboCider','python':sys.version,'requirements':lock.read_text()},indent=2)+'\n')
        print('Ready:',destination,flush=True)
if __name__=='__main__':main()
