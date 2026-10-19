#!/usr/bin/env python3
"""Seal an already collected tool snapshot without changing any measured source."""
import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tarfile


def seal(root,output):
    summary=json.loads((root/'summary.json').read_text());source=json.loads((root/'source.json').read_text())
    if summary['status']!='complete' or set(source)!=set(summary['source_sha256']):raise ValueError('complete bound collected source snapshot required')
    payload=[]
    for name,value in source.items():
        p=PurePosixPath(name);raw=value.encode()
        if p.is_absolute() or '..' in p.parts or not name.startswith('tools/') or hashlib.sha256(raw).hexdigest()!=summary['source_sha256'][name]:
            raise ValueError('safe complete source snapshot and matching digest required')
        payload.append((name,raw))
    # x:gz cannot replace an old archive. This creates an artifact, not source
    # files or a path restoration, and never executes archived code.
    with tarfile.open(output,'x:gz') as archive:
        for name,raw in sorted(payload):
            member=tarfile.TarInfo(name);member.size=len(raw);member.mode=0o644;member.mtime=0;archive.addfile(member,io.BytesIO(raw))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    seal(args.cohort,args.output)
