#!/usr/bin/env python3
"""Retire only the explicitly listed obsolete Oct11-13 experimental exports.

Authorized for disk recovery by the user on 2026-10-19. Original models,
references, current Qwen experiments and all results are deliberately excluded.
Small metadata/source files are retained in an archive, large derived payloads
are reproducible but not recoverable from this archive.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
TARGETS=(
    'outputs/benchmark-model-views/qwen21-bf16-mflux-v2-complete-20261011',
    'outputs/flux4-single40-bf16-derived-v1-20261013',
    'outputs/flux4-native-saved-q4-view-v3-20261012',
    'outputs/flux9-ane-prefix4096-r1536-compiled-v1-20261011',
    'outputs/flux9-ane-prefix4096-r1536-export-v1-20261011',
    'outputs/flux4-single40-ane-compiled-v1-20261013',
    'outputs/flux4-single40-ane-a6144-b1536-v1-20261013',
)
LINK_TARGETS={
    TARGETS[0]+'/processor/tokenizer.json':'models/Comfy-Org-Qwen-Image-2.1/processor/tokenizer.json',
    TARGETS[2]+'/tokenizer':'models/FLUX.2-klein-4B-mflux-Q4/tokenizer',
    TARGETS[1]+'/text_encoder':TARGETS[2]+'/text_encoder',
    TARGETS[1]+'/tokenizer':'models/FLUX.2-klein-4B-mflux-Q4/tokenizer',
    TARGETS[1]+'/vae':TARGETS[2]+'/vae',
}


def inspect():
    rows=[];small=[]
    for name in TARGETS:
        p=ROOT/name
        if not p.is_dir() or p.is_symlink() or p.resolve()!=p or len(p.relative_to(ROOT).parts)<2:raise ValueError('exact existing non-symlink experimental subtree required: '+name)
        opened=subprocess.run(['lsof','-n','-P','+D',str(p),'-F','pn'],capture_output=True,text=True)
        if opened.returncode!=1 or opened.stdout.strip() or opened.stderr.strip():raise ValueError('open handles or incomplete lsof check; refusing '+name)
        entries=[]
        for f in sorted(p.rglob('*')):
            if f.is_symlink():
                relative=str(f.relative_to(ROOT));expected=LINK_TARGETS.get(relative)
                if expected is None or f.resolve(strict=True)!=(ROOT/expected).resolve(strict=True):raise ValueError('uninspected companion link: '+relative)
                st=f.lstat();entries.append(dict(path=relative,bytes=st.st_size,allocated_bytes=st.st_blocks*512,nlink=st.st_nlink,metadata_retained=True,
                    symlink_target=expected,policy='archive/unlink link only; never follow or remove original target'));small.append(f);continue
            if f.is_file():
                st=f.stat();relative=str(f.relative_to(ROOT));retained=st.st_size<=1<<20
                entries.append(dict(path=relative,bytes=st.st_size,allocated_bytes=st.st_blocks*512,nlink=st.st_nlink,metadata_retained=retained))
                if retained:small.append(f)
        rows.append(dict(path=name,files=entries,logical_bytes=sum(x['bytes'] for x in entries),allocated_bytes_before_sharing=sum(x['allocated_bytes'] for x in entries)))
    return rows,small


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');args=p.parse_args();rows,small=inspect()
    print(json.dumps(dict(targets=[dict(path=r['path'],logical_bytes=r['logical_bytes']) for r in rows],metadata_files=len(small),apply=args.apply),indent=2),flush=True)
    if not args.apply:return
    archive=ROOT/'outputs/qwen-expanded-retired-exports-metadata-v1-20261019.tar.gz';receipt=ROOT/'outputs/qwen-expanded-retired-exports-v1-20261019.json'
    if archive.exists() or receipt.exists():raise ValueError('fresh cleanup artifacts required')
    before=shutil.disk_usage(ROOT).free
    with tarfile.open(archive,'x:gz',dereference=False) as tf:
        for f in small:tf.add(f,arcname=str(f.relative_to(ROOT)),recursive=False)
    record=dict(schema='tc-obsolete-derived-exports-retirement-v1',status='archived_before_deletion',free_before=before,targets=rows,
        metadata_archive=str(archive.relative_to(ROOT)),original_models_untouched=True,current_qwen_evidence_untouched=True,
        recovery='small files archived; large derived payloads require regeneration from original models, not recoverable from archive')
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    for row in rows:
        target=ROOT/row['path']
        if target.is_symlink() or target.resolve()!=target or target.relative_to(ROOT).as_posix() not in TARGETS:raise ValueError('retirement target changed')
        shutil.rmtree(target)
    record.update(status='complete',free_after=shutil.disk_usage(ROOT).free)
    record['observed_free_delta_bytes']=record['free_after']-before;receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(status='complete',observed_free_delta_bytes=record['observed_free_delta_bytes'],free_after=record['free_after'])),flush=True)


if __name__=='__main__':main()
