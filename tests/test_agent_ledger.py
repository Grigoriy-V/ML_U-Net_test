import json, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import agent_ledger as al

def meta(run='run-1'):
    return {'agent_run_id':run,'parent_task':'/root','agent_name':'terra_worker','requested_model':'gpt-5.6-terra','requested_reasoning':'low','task_type':'test','roadmap_step':'step','scope_summary':'test scope','constraints':['bounded'],'commands':['test'],'files_changed':[],'git_commit_before':None,'git_commit_after':None,'ml_ledger_event_ids':[],'notes':'test'}

class LedgerTests(unittest.TestCase):
  def setUp(self): self.tmp=tempfile.TemporaryDirectory(); self.ledger=Path(self.tmp.name)/'ledger.jsonl'
  def tearDown(self): self.tmp.cleanup()
  def add_start(self, run='run-1'):
    event=al.start_from_metadata(meta(run)); al.append_event(self.ledger,event); return event
  def test_lifecycle_completed_reviewed(self):
    self.add_start(); al.append_event(self.ledger,al.inherited_event(al.read_events(self.ledger),'run-1','completed',outcome_summary='ok',files_changed=['report.md'],commands=['test'],duration_seconds=1,notes='agent_ledger.py'))
    al.append_event(self.ledger,al.inherited_event(al.read_events(self.ledger),'run-1','reviewed',supervisor_decision='accept',outcome_summary='accepted',notes='agent_ledger.py'))
    self.assertEqual(len(al.read_events(self.ledger)),3)
  def test_terminal_requires_actual_evidence_and_computes_duration(self):
    self.add_start()
    self.assertEqual(al.main(['--ledger',str(self.ledger),'terminal','--run-id','run-1','--status','completed','--outcome-summary','x','--files-changed-json','[]','--commands-json','[]']),2)
    self.assertEqual(len(al.read_events(self.ledger)),1)
    self.assertEqual(al.main(['--ledger',str(self.ledger),'terminal','--run-id','run-1','--status','completed','--outcome-summary','x','--files-changed-json','[]','--commands-json','["pytest"]']),0)
    self.assertGreater(al.read_events(self.ledger)[-1]['duration_seconds'],0)
  def test_review_requires_terminal_and_uses_reviewer_identity(self):
    self.add_start()
    args=['--ledger',str(self.ledger),'review','--run-id','run-1','--decision','accept','--outcome-summary','x','--reviewer-agent-name','root_supervisor','--reviewer-model','root-session-model','--reviewer-reasoning','not_applicable','--parent-task','/root']
    self.assertEqual(al.main(args),2)
    al.append_event(self.ledger,al.inherited_event(al.read_events(self.ledger),'run-1','completed',outcome_summary='x',files_changed=[],commands=['x'],duration_seconds=1,notes='agent_ledger.py'))
    self.assertEqual(al.main(args),0)
    review=al.read_events(self.ledger)[-1]; self.assertEqual(review['agent_name'],'root_supervisor'); self.assertEqual(review['requested_model'],'root-session-model')
  def test_failed_and_interrupted(self):
    for run,status in [('a','failed'),('b','interrupted')]:
      self.add_start(run); al.append_event(self.ledger,al.inherited_event(al.read_events(self.ledger),run,status,outcome_summary=status,files_changed=[],commands=['test'],duration_seconds=0,notes='agent_ledger.py'))
  def test_dry_run_and_invalid_do_not_mutate(self):
    event=al.start_from_metadata(meta()); al.append_event(self.ledger,event,dry_run=True); self.assertFalse(self.ledger.exists())
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,{'bad':True})
    self.assertFalse(self.ledger.exists())
  def test_absolute_and_worker_decision_rejected(self):
    event=al.start_from_metadata(meta()); event['files_changed']=['C:/bad'];
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,event)
    event=al.start_from_metadata(meta()); event['supervisor_decision']='accept'
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,event)
  def test_missing_start_duplicate_terminal_and_review_rejected(self):
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,al.inherited_event([], 'none','completed',outcome_summary='x'))
    self.add_start(); done=al.inherited_event(al.read_events(self.ledger),'run-1','completed',outcome_summary='x',files_changed=[],commands=['x'],duration_seconds=0,notes='agent_ledger.py'); al.append_event(self.ledger,done)
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,al.inherited_event(al.read_events(self.ledger),'run-1','failed',outcome_summary='x',files_changed=[],commands=['x'],duration_seconds=0,notes='agent_ledger.py'))
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,al.inherited_event([self.add_start('other')],'other','reviewed',supervisor_decision='accept',outcome_summary='x',notes='agent_ledger.py'))
  def test_duplicate_id_correction_and_eof(self):
    first=self.add_start(); original=self.ledger.read_bytes(); correction=al.start_from_metadata(meta('fix')); correction.update({'event_type':'correction','status':'corrected','supervisor_decision':None}); al.append_event(self.ledger,correction)
    self.assertTrue(self.ledger.read_bytes().startswith(original))
    duplicate=al.start_from_metadata(meta('other')); duplicate['event_id']=first['event_id']
    with self.assertRaises(al.LedgerError): al.append_event(self.ledger,duplicate)
  def test_concurrent_writers_no_corruption(self):
    script="from pathlib import Path; import sys; sys.path.insert(0, sys.argv[2]); import agent_ledger as a; e=a.start_from_metadata({'agent_run_id':sys.argv[3],'parent_task':'/root','agent_name':'t','requested_model':'m','requested_reasoning':'low','task_type':'t','roadmap_step':'s','scope_summary':'s','constraints':[],'commands':[],'files_changed':[],'git_commit_before':None,'git_commit_after':None,'ml_ledger_event_ids':[],'notes':'agent_ledger.py'}); a.append_event(Path(sys.argv[1]),e)"
    procs=[subprocess.Popen([sys.executable,'-c',script,str(self.ledger),str(ROOT/'tools'),f'r{i}']) for i in range(4)]
    self.assertTrue(all(p.wait()==0 for p in procs)); self.assertEqual(len(al.read_events(self.ledger)),4)
  def test_windows_lock_failure_empty_ledger_has_no_byte_mutation(self):
    self.ledger.touch(); before=self.ledger.read_bytes()
    failing=types.SimpleNamespace(LK_LOCK=1,LK_UNLCK=2,locking=lambda *args: (_ for _ in ()).throw(OSError('lock failed')))
    with mock.patch.object(al.os,'name','nt'), mock.patch.dict(sys.modules,{'msvcrt':failing}):
      with self.assertRaises(OSError):
        with al.locked_append(self.ledger): pass
    self.assertEqual(self.ledger.read_bytes(),before)
  def test_production_legacy_warning(self):
    warnings=al.lifecycle(al.read_events(ROOT/'reports'/'agent_execution_ledger.jsonl'))
    self.assertTrue(any('legacy warning' in x for x in warnings))

if __name__ == '__main__': unittest.main()
