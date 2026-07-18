import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));import validate_core_pin as v
class PinTests(unittest.TestCase):
 def lock(self): return json.loads((ROOT/'orchestration.lock.json').read_text())
 def test_valid_local(self): v.validate(self.lock(),ROOT)
 def test_bad_constraints(self):
  for change in ('missing','extra','duplicate','relation','hash','path'):
   x=self.lock()
   if change=='missing': x.pop('core_version')
   if change=='extra': x['x']=1
   if change=='duplicate': x['managed_files'].append(x['managed_files'][0].copy())
   if change=='relation': x['managed_files'][0]['relationship']='bad'
   if change=='hash': x['managed_files'][0]['source_sha256']='0'*64
   if change=='path': x['managed_files'][0]['source_path']='../bad'
   with self.assertRaises(ValueError): v.validate(x,ROOT)
 def test_core_valid_and_wrong(self):
  core=Path('D:/ML/human-in-the-loop-ml-orchestration');v.validate(self.lock(),ROOT,core);x=self.lock();x['core_commit']='0'*40
  with self.assertRaises(ValueError):v.validate(x,ROOT,core)
 def test_linked_paths_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); external=d/'outside';external.write_text('x')
   link=ROOT/'tests'/'_pin_link_tmp'
   try:
    link.symlink_to(external)
   except (OSError,NotImplementedError): self.skipTest('symlink fixture unavailable')
   try:
    x=self.lock();x['managed_files'][0]['source_path']='tests/_pin_link_tmp'
    with self.assertRaises(ValueError):v.validate(x,ROOT)
   finally: link.unlink(missing_ok=True)
if __name__=='__main__':unittest.main()
