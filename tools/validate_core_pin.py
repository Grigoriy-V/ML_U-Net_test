import argparse, hashlib, json, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; REQUIRED={'AGENTS.md','.codex/config.toml','.codex/agents/luna_clerk.toml','.codex/agents/terra_worker.toml','.codex/agents/sol_specialist.toml','tools/agent_ledger.py','reports/agent_execution_ledger.schema.json','docs/agent_orchestration.md'}
def h(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def safe(x): return isinstance(x,str) and not Path(x).is_absolute() and '..' not in Path(x).parts and not Path(x).drive
def reparse(p):
 try:
  st=p.lstat()
  return p.is_symlink() or bool(getattr(st,'st_file_attributes',0)&0x400)
 except OSError as e: raise ValueError('cannot establish link status') from e
def checked(root,relative):
 if not safe(relative): raise ValueError('unsafe path')
 root=root.resolve(strict=True); candidate=root.joinpath(*Path(relative).parts)
 try: candidate.relative_to(root)
 except ValueError: raise ValueError('path escapes root')
 current=root
 for part in Path(relative).parts:
  if reparse(current): raise ValueError('linked/reparse parent')
  current=current/part
 if reparse(current): raise ValueError('linked/reparse file')
 try: current.resolve(strict=True).relative_to(root)
 except (OSError,ValueError): raise ValueError('resolved path escapes root')
 return current
def validate(lock,root,core=None):
 if set(lock)!={'schema_version','core_repository','core_commit','core_version','adapter_type','managed_files'} or lock['schema_version']!='1.0' or not isinstance(lock['managed_files'],list): raise ValueError('invalid lock structure')
 seen=set()
 for e in lock['managed_files']:
  if set(e)!={'source_path','source_sha256','core_path','core_sha256','relationship'} or e['relationship'] not in {'exact_copy','adapted','project_override'}: raise ValueError('invalid entry')
  if not safe(e['source_path']) or not safe(e['core_path']) or e['source_path'] in seen: raise ValueError('invalid or duplicate path')
  seen.add(e['source_path'])
  if not all(isinstance(e[k],str) and len(e[k])==64 and all(c in '0123456789abcdef' for c in e[k]) for k in ('source_sha256','core_sha256')): raise ValueError('invalid hash')
  source=checked(root,e['source_path'])
  if not source.is_file() or h(source)!=e['source_sha256']: raise ValueError('source hash mismatch')
  if core:
   target=checked(core,e['core_path'])
   if not target.is_file() or h(target)!=e['core_sha256']: raise ValueError('core hash mismatch')
   if e['relationship']=='exact_copy' and e['source_sha256']!=e['core_sha256']: raise ValueError('exact copy mismatch')
 if seen!=REQUIRED: raise ValueError('managed coverage mismatch')
 if core:
  if (core/'VERSION').read_text().strip()!=lock['core_version']: raise ValueError('core version mismatch')
  got=subprocess.check_output(['git','-C',str(core),'rev-parse','HEAD'],text=True).strip()
  if got!=lock['core_commit']: raise ValueError('core commit mismatch')
def main():
 p=argparse.ArgumentParser();p.add_argument('--core-root',type=Path);a=p.parse_args();
 try: validate(json.loads((ROOT/'orchestration.lock.json').read_text()),ROOT,a.core_root);print('valid: core pin')
 except Exception as e: print('error:',e);return 2
 return 0
if __name__=='__main__': raise SystemExit(main())
