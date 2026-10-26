"""Fresh-process collection with integrity stops; never chooses reports by quality."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from .recovery import audit, IDS, digest, read

VERSION='adaptive-depth-sequential-reliability-v1'
ROOT=Path('runs/adaptive-depth-sequential-reliability-v1')
def now():return datetime.now(timezone.utc).isoformat()
def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

def verify():
    manifest=read('configs/adaptive_depth_reliability_v1.json')
    for p,h in manifest['hashes'].items():
        if digest(p)!=h:raise ValueError(f'Reliability freeze changed: {p}')
    base=yaml.safe_load(Path('configs/adaptive_depth_recovery.yaml').read_text())
    for q in IDS:
        c=yaml.safe_load(Path(f'configs/adaptive_depth_sequential_{q}.yaml').read_text())
        assert c['dataset']['query_ids']==[q] and c['dataset']['limit']==1
        assert {k:v for k,v in c['dataset'].items() if k not in ['query_ids','limit']}=={k:v for k,v in base['dataset'].items() if k not in ['query_ids','limit']}
        for key in ['run_name','output_dir','dataset']:c.pop(key)
        b={k:v for k,v in base.items() if k not in ['run_name','output_dir','dataset']}
        assert c==b
    return manifest

def execute(config,attempt_dir):
    """Wait for a fresh interpreter, preserving only redacted detailed diagnostics."""
    from .diagnostics import Diagnostics,VERSION as diagnostics_version
    command=[sys.executable,'-m','adr.cli','run','--config',str(config)]
    started=now()
    env=dict(os.environ);env['PYTHONPATH']='src';env['PYTHONDONTWRITEBYTECODE']='1'
    log_path=attempt_dir/'console.redacted.jsonl'
    diagnostics=Diagnostics(log_path,env)
    proc=None;code=None
    try:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',env=env)
        write(attempt_dir/'process.json',{'pid':proc.pid,'started_at':started,'command':command,'ended_at':None,'diagnostics_version':diagnostics_version,'diagnostics_path':str(log_path)})
        for line in proc.stdout:diagnostics.record(line)
        code=proc.wait()
    except BaseException:
        if proc is not None:
            proc.terminate();proc.wait()
        raise
    finally:
        if proc is not None:proc.stdout.close()
        diagnostics.close()
        write(attempt_dir/'process.json',{'pid':None if proc is None else proc.pid,'started_at':started,'ended_at':now(),'command':command,'returncode':code,'error_categories':sorted(diagnostics.blocking),'diagnostic_categories':sorted(diagnostics.categories),'diagnostics_version':diagnostics_version,'diagnostics_path':str(log_path)})
    return code,sorted(diagnostics.blocking)

def compatible_manifest(old,new):
    if old==new:return True
    previous=hashlib.sha256(json.dumps(old,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return previous==new.get('diagnostics_predecessor_manifest_sha256')

def collect(root=ROOT,retry_failed=False,runner=execute):
    manifest=verify();root=Path(root);root.mkdir(parents=True,exist_ok=True)
    # Exclusive ownership prevents two wrappers from starting overlapping queries.
    lock=root/'collection.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        path=root/'collection.json';state=read(path) if path.exists() else {'version':VERSION,'manifest':manifest,'queries':{}}
        if not compatible_manifest(state['manifest'],manifest):raise ValueError('Collection freeze differs; do not mix revisions')
        if state['manifest']!=manifest:
            state.setdefault('manifest_history',[]).append(state['manifest']);state['manifest']=manifest
            write(path,state)  # diagnostic-only revision; never changes acceptance status
        for q in IDS:
            records=state['queries'].setdefault(q,[])
            if records and records[-1]['status']=='approved':
                last=records[-1]
                if any(digest(p)!=h for p,h in last['hashes'].items()):raise ValueError('Accepted source modified')
                continue
            if records and not retry_failed:raise ValueError(f'q{q} unfinished/failed; inspect then explicitly use --retry-failed')
            a=root/f'q{q}'/f'attempt-{len(records)+1:03}';a.mkdir(parents=True,exist_ok=False)
            c=yaml.safe_load(Path(f'configs/adaptive_depth_sequential_{q}.yaml').read_text());c['output_dir']=str(a/'research')
            cfg=a/'config.yaml';cfg.write_text(yaml.safe_dump(c,sort_keys=False))
            rec={'status':'running','started_at':now(),'attempt_directory':str(a)};records.append(rec);write(path,state)
            print(json.dumps({'query':q,'status':'starting','attempt':len(records)}),flush=True)
            import runpy
            runpy.run_path('scripts/restore_cl100k_cache.py')['restore']()
            code,signals=runner(cfg,a)
            dirs=list((a/'research').glob('*/metrics/summary.json'))
            integrity=None
            if len(dirs)==1:
                run=dirs[0].parents[1];rec['run']=str(run)
                try:integrity=audit(run,expected_ids=[q])
                except Exception as exc:integrity={'flags':['audit_failed:'+type(exc).__name__]}
            rec.update(ended_at=now(),returncode=code,error_categories=signals,integrity=integrity)
            okay=code==0 and integrity is not None and not integrity['flags'] and not signals
            rec['status']='approved' if okay else 'failed_requires_inspection'
            if okay:rec['hashes']={str(p):digest(p) for p in run.rglob('*') if p.is_file()}
            write(path,state);write(a/'integrity.json',integrity)
            print(json.dumps({'query':q,'status':rec['status'],'run':rec.get('run'),'error_categories':signals,'flags':None if integrity is None else integrity['flags']}),flush=True)
            if not okay:return False
        return True
    finally:lock.unlink()

def aggregate(root,output):
    manifest=verify();state=read(Path(root)/'collection.json');output=Path(output)
    if not compatible_manifest(state['manifest'],manifest):raise ValueError('Mixed reliability revision')
    if set(state['queries'])!=set(IDS):raise ValueError('Missing/unexpected IDs')
    selected=[]
    for q in IDS:
        records=state['queries'][q];approved=[r for r in records if r['status']=='approved']
        if len(approved)!=1 or records[-1] is not approved[0]:raise ValueError('Missing/duplicate approvals or unresolved newer attempt')
        a=approved[0]
        if a['error_categories'] or any(digest(p)!=h for p,h in a['hashes'].items()):raise ValueError('Source changed or error signals')
        r=Path(a['run']);check=audit(r,expected_ids=[q])
        if check['flags']:raise ValueError('Source integrity no longer valid')
        selected.append((q,r))
    if output.exists():raise ValueError('Output already exists')
    import shutil
    from adr.core.types import Trajectory
    from adr.eval.local_metrics import compute_local_metrics
    trajectories=[]
    for q,r in selected:
        t=Trajectory.model_validate(read(r/'trajectories'/f'{q}.json'))
        if str(t.query.id)!=q:raise ValueError('Trajectory ID mismatch')
        trajectories.append(t)
    metrics=compute_local_metrics(trajectories)
    output.mkdir(parents=True)
    for q,r in selected:
        for folder,suffix in [('trajectories','.json'),('reports','.md'),('queries','.json')]:
            (output/folder).mkdir(exist_ok=True);shutil.copyfile(r/folder/f'{q}{suffix}',output/folder/f'{q}{suffix}')
        (output/'branch_expansion').mkdir(exist_ok=True)
        for p in (r/'branch_expansion').glob('*.jsonl'):shutil.copyfile(p,output/'branch_expansion'/p.name)
    (output/'metrics').mkdir();write(output/'metrics/local.json',metrics)
    write(output/'metrics/summary.json',{k:v for k,v in metrics.items() if k!='per_query'}|{'run_id':output.name,'scores':{},'official':{}})
    write(output/'aggregation_provenance.json',{'version':VERSION,'collection':state,'selected':[(q,str(r)) for q,r in selected],'Q_per_second':None,'quality_selection':False})
    write(output/'source_checksums.json',{p:h for q in IDS for a in state['queries'][q] if a['status']=='approved' for p,h in a['hashes'].items()})
    return str(output)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('collect');a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--retry-failed',action='store_true')
    a=sub.add_parser('aggregate');a.add_argument('--root',type=Path,default=ROOT);a.add_argument('--output',type=Path,required=True)
    sub.add_parser('verify');args=p.parse_args()
    if args.cmd=='verify':print(json.dumps(verify(),indent=2))
    elif args.cmd=='collect':
        if not collect(args.root,args.retry_failed):raise SystemExit(1)
    else:print(aggregate(args.root,args.output))
if __name__=='__main__':main()
