#!/usr/bin/env python3
"""Run official-IDQL numerical checks on two virtual CPU devices only.

This verifies source identity and numerical execution, not CUDA readiness,
robot rollout performance, or the equivalence of unspecified image features.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
REQUIRED_CORE_CASES={
 'test_fixed_source_bytes_and_real_bound_function_objects',
 'test_default_vector_network_initialization_and_forward_match_official_create',
 'test_five_updates_match_independent_full_official_learner_params_adam_and_ema',
 'test_ema_updates_once_and_value_target_uses_minimum_of_both_target_qs',
 'test_two_virtual_cpu_pmap_reduces_gradients_before_adam_vs_global64',
 'test_target_q_min_and_greedy_match_upstream_compute_q_for_candidates',
 'test_full_checkpoint_restore_preserves_all_state_and_numpy_sampler',
 'test_explicit_two_camera_upstream_pixel_adapter_cpu_smoke',
}
REQUIRED_TEST_FILES={f'test_official_idql_{name}.py' for name in
 ['core','augmentation','runner','diagnostics','adapters','scoring','selection']}
COLLECTION_PLUGIN='''
import json, os
from pathlib import Path
collected=[]
executed=[]
deselected=[]
def pytest_itemcollected(item):
 collected.append(item.nodeid)
def pytest_deselected(items):
 deselected.extend(item.nodeid for item in items)
def pytest_runtest_logreport(report):
 if report.when=='call':executed.append(report.nodeid)
def pytest_sessionfinish(session,exitstatus):
 Path(os.environ['OFFICIAL_IDQL_COLLECTION_AUDIT_PATH']).write_text(json.dumps({
  'collected':collected,'selected':[item.nodeid for item in session.items],
  'executed':executed,'deselected':deselected,'exitstatus':int(exitstatus)},indent=2)+'\\n')
'''

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_files():
 # Bind common replay/data/IO dependencies and test helpers as well as the new
 # modules. The manifest alone cannot detect a late edit to a vendor payload.
 files=list((ROOT/'src').rglob('*.py'))
 files.extend([ROOT/'src/post_training/_vendor/idql_official/UPSTREAM.json',Path(__file__).resolve()])
 files.extend((ROOT/'tests').rglob('*.py'))
 vendor=ROOT/'src/post_training/_vendor/idql_official'
 manifest=json.loads((vendor/'UPSTREAM.json').read_text())
 files.extend(vendor/name for name in manifest['files'])
 for config in [ROOT/'pyproject.toml',ROOT/'pytest.ini',ROOT/'setup.cfg',ROOT/'requirements.txt']:
  if config.exists():files.append(config)
 if (ROOT/'conftest.py').exists():files.append(ROOT/'conftest.py')
 for entry in [ROOT/'train_idql_official.py',ROOT/'scripts/plot_official_idql_metrics.py']:
  if entry.exists():files.append(entry)
 files.extend((ROOT/'configs/official_idql').glob('*.yaml'))
 return {str(path.relative_to(ROOT)):digest(path) for path in sorted(set(files))}


def parse_affinity(value):
 result=set()
 for item in value.split(','):
  if '-' in item:
   first,last=(int(x) for x in item.split('-',1))
   if first>last:raise ValueError('CPU affinity range is reversed')
   result.update(range(first,last+1))
  else:result.add(int(item))
 if not result or min(result)<0:raise ValueError('CPU affinity must contain nonnegative CPU indices')
 return result

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--output',type=Path,default=ROOT/'reports/official_idql_cpu_verification.json')
 p.add_argument('--cpu-affinity',help='Optional existing CPU subset, e.g.0-7 or32,34; default respects the current cpuset')
 args=p.parse_args(argv);output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
 if hasattr(os,'sched_getaffinity'):
  allowed=os.sched_getaffinity(0)
  requested=parse_affinity(args.cpu_affinity) if args.cpu_affinity else allowed
  if not requested<=allowed:raise RuntimeError('Requested CPU affinity is outside this process allowed cpuset')
  affinity=sorted(requested)
  if args.cpu_affinity:os.sched_setaffinity(0,affinity)
 else:affinity=None
 # Environment is set before the child can import JAX or any model module.
 env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu',CUDA_VISIBLE_DEVICES='',
  JAX_ENABLE_X64='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for key in ['PYTEST_ADDOPTS','PYTEST_PLUGINS']:env.pop(key,None)
 env['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
 flags=env.get('XLA_FLAGS','').split()
 flags=[v for v in flags if not v.startswith('--xla_force_host_platform_device_count=')]
 env['XLA_FLAGS']=' '.join(flags+['--xla_force_host_platform_device_count=2'])
 env['PYTHONPATH']=str(ROOT/'src')+os.pathsep+env.get('PYTHONPATH','')
 with tempfile.TemporaryDirectory(prefix='official-idql-cpu-') as directory:
  sources_before=source_files()
  test_files=sorted((ROOT/'tests').glob('test_official_idql_*.py'))
  missing_files=REQUIRED_TEST_FILES-{p.name for p in test_files}
  if missing_files:raise RuntimeError('Required official-IDQL CPU suites missing: '+', '.join(sorted(missing_files)))
  temp=Path(directory);metrics=temp/'measurements.json';junit=temp/'junit.xml';audit_path=temp/'collection.json'
  (temp/'official_idql_collection_audit.py').write_text(COLLECTION_PLUGIN)
  env['PYTHONPATH']=str(temp)+os.pathsep+env['PYTHONPATH']
  env['OFFICIAL_IDQL_COLLECTION_AUDIT_PATH']=str(audit_path)
  env['OFFICIAL_IDQL_TEST_METRICS_PATH']=str(metrics)
  command=[sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','-p','official_idql_collection_audit',
   '-o','addopts=','--rootdir='+str(ROOT),'--confcutdir='+str(ROOT),*[str(path) for path in test_files],
   '--junitxml='+str(junit)]
  result=subprocess.run(command,cwd=ROOT,env=env,text=True,capture_output=True)
  sources_after=source_files()
  measured=json.loads(metrics.read_text()) if metrics.exists() else {}
  audit=json.loads(audit_path.read_text()) if audit_path.exists() else {}
  suites=ET.parse(junit).getroot() if junit.exists() else None
  counts={name:sum(int(s.get(name,0)) for s in suites.iter('testsuite')) if suites is not None else None
   for name in ['tests','errors','failures','skipped']}
  executed_core={case.get('name') for case in suites.iter('testcase')
   if 'test_official_idql_core' in case.get('classname','') and case.find('skipped') is None} if suites is not None else set()
  missing_cases=REQUIRED_CORE_CASES-executed_core
  passed=result.returncode==0 and bool(measured) and bool(counts['tests']) and counts['errors']==counts['failures']==0
  # Fail closed when execution silently skipped any required numerical test.
  collected=audit.get('collected',[]);executed=audit.get('executed',[])
  collection_complete=(bool(collected) and len(collected)==len(set(collected))
   and len(executed)==len(set(executed)) and audit.get('deselected')==[]
   and set(audit.get('selected',[]))==set(collected)==set(executed)
   and len(collected)==counts['tests'])
  passed=(passed and counts['skipped']==0 and counts['tests']>=30 and not missing_cases
   and collection_complete and sources_before==sources_after)
  report={'schema_version':1,'status':'PASS_OFFICIAL_IDQL_SOURCE_AND_NUMERICS_CPU_ONLY' if passed else 'FAIL_OFFICIAL_IDQL_CPU_VERIFICATION',
   'observed_at':datetime.now(timezone.utc).isoformat(),'python_executable':sys.executable,'python_version':sys.version,
   'source_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip(),
   'test_counts':counts,'command':command,'returncode':result.returncode,'measurements':measured,
   'selected_test_files':[str(path.relative_to(ROOT)) for path in test_files],
   'collection_audit':audit,'all_collected_cases_executed_exactly_once':collection_complete,
   'inherited_pytest_options_ignored':True,'third_party_pytest_plugin_autoload':False,
   'required_core_cases_executed':sorted(REQUIRED_CORE_CASES & executed_core),'missing_required_core_cases':sorted(missing_cases),
   'cpu_affinity':affinity,
   'source_files_sha256':sources_after,'source_files_unchanged_during_verification':sources_before==sources_after,
   'test_source_sha256':digest(ROOT/'tests/test_official_idql_core.py'),'verification_tool_sha256':digest(Path(__file__)),
   'environment':{k:env[k] for k in ['JAX_PLATFORMS','JAX_PLATFORM_NAME','CUDA_VISIBLE_DEVICES','XLA_FLAGS','JAX_ENABLE_X64']},
   'stdout':result.stdout,'stderr':result.stderr,'gpu_initialized':False,'gpu_numerical_equivalence_claimed':False,
   'production_training_or_platform_evaluation_started':False,
   'synthetic_cpu_test_parameter_updates_executed':bool(measured.get('five_updates_vs_official_full_creator')),
   'production_data_or_checkpoint_gate_pass_claimed':False,
   'limitations':['CPU fixtures do not establish that a production critic passes held-out quality gates.',
    'CPU vectors do not validate any unconfigured visual feature extractor.',
    'Virtual CPU pmap does not establish CUDA performance or GPU numerical trajectories.',
    'Reference construction uses Gym Box spaces only; this environment is not a reproduction of the upstream MuJoCo dependency stack.']}
  temporary=output.with_name(output.name+f'.{os.getpid()}.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temporary,output)
  print(json.dumps({'status':report['status'],'receipt':str(output),'receipt_sha256':digest(output),'test_counts':counts,'returncode':result.returncode}),flush=True)
  if result.returncode:print(result.stdout+result.stderr,file=sys.stderr)
  return 0 if passed else 1

if __name__=='__main__':raise SystemExit(main())
