"""Remove only this campaign's owned artifacts and newly created executable caches."""
import json,shutil,os,time,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# All five names were created in this campaign (birth times preserved below).
paths=[ROOT/'cache']+[Path.home()/'Library/Caches'/name for name in ['transformer','transformer-heads','transformer-private','transformer-private-gpu','transformer-startup']]
before=shutil.disk_usage(ROOT).free;items=[]
for p in paths:
    if not p.exists():continue
    birth=p.stat().st_birthtime
    if p.parent==Path.home()/'Library/Caches' and birth<1788970359:
        raise RuntimeError(f'refuse preexisting cache {p}')
    size=sum(x.stat().st_size for x in p.rglob('*') if x.is_file() and not x.is_symlink())
    items.append({'path':str(p),'logical_bytes':size,'birth_utc':datetime.datetime.fromtimestamp(birth,datetime.timezone.utc).isoformat()})
    shutil.rmtree(p)
    items[-1]['exists_after']=p.exists()
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'final_cleanup':items,'disk_free_before':before,'disk_free_after':shutil.disk_usage(ROOT).free,'scope':'Only task-owned cache tree and five executable cache directories born during this campaign. Existing global Core ML/other application caches preserved.'}
(ROOT/'notes/raw/final_cleanup.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
