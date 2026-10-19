#!/usr/bin/env python3
"""Create a byte-verified, exact-manifest source archive without AppleDouble extras.

Can reseal source bytes from an older archive after the working tree changed.
The original archive is never modified, extracted, deleted or executed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[2]


def seal(manifest,output,source_archive=None):
    entries=manifest['inputs']['sources'];expected={x['path'] for x in entries}
    if len(expected)!=len(entries):raise ValueError('unique complete source manifest required')
    source=tarfile.open(source_archive,'r:gz') if source_archive else None
    try:
        members={m.name:m for m in source.getmembers()} if source else {}
        if source and not expected<=set(members):raise ValueError('source archive is incomplete')
        payload=[]
        for e in entries:
            name=e['path']
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('bounded repository-relative source path required')
            if source:
                m=members[name]
                if not m.isfile():raise ValueError('source file must contain actual regular bytes')
                raw=source.extractfile(m).read();mode=m.mode
            else:raw=(ROOT/name).read_bytes();mode=(ROOT/name).stat().st_mode&0o777
            if hashlib.sha256(raw).hexdigest()!=e['sha256']:raise ValueError('source does not match actual library manifest: '+name)
            payload.append((name,raw,mode))
        with tarfile.open(output,'x:gz') as tf:
            for name,raw,mode in sorted(payload):
                member=tarfile.TarInfo(name);member.size=len(raw);member.mode=mode;member.mtime=0;tf.addfile(member,io.BytesIO(raw))
        return dict(files=len(payload),source_archive_extras_preserved_in_original=len(set(members)-expected) if source else 0)
    finally:
        if source:source.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--from-archive',type=Path);a=p.parse_args();print(json.dumps(seal(json.loads(a.manifest.read_text()),a.output,a.from_archive)))
