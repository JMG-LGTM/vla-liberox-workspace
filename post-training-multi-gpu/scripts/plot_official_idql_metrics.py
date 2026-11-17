#!/usr/bin/env python3
"""Render raw and EWMA Q/V losses from explicit CSV snapshots, on CPU only.

Training rows must contain every optimizer update, exactly once, in order.
Validation is sparse and never interpolated. Invalid or partial input fails
before replacing plots. Neither input CSV is modified.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

LOSSES=('value_loss','critic_loss')

def sha(data):return hashlib.sha256(data).hexdigest()

def read_snapshot(path,*,sparse=False):
 path=Path(path);raw=path.read_bytes()
 if not raw or not raw.endswith(b'\n'):raise ValueError(f'{path}: empty or partially written CSV; retry after a complete row is flushed')
 reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig'),newline=''))
 fields=reader.fieldnames
 if not fields or len(fields)!=len(set(fields)):raise ValueError(f'{path}: missing or duplicate header')
 required={'optimizer_update',*LOSSES}
 if not required<=set(fields):raise ValueError(f'{path}: required columns missing: {sorted(required-set(fields))}')
 rows=list(reader)
 if not rows:raise ValueError(f'{path}: no recorded updates')
 updates=[];values={key:[] for key in LOSSES}
 for line,row in enumerate(rows,2):
  if None in row or any(value is None for value in row.values()):raise ValueError(f'{path}:{line}: partial or malformed row')
  token=row['optimizer_update']
  if not token.isdecimal() or int(token)<1:raise ValueError(f'{path}:{line}: optimizer_update must be a positive integer')
  step=int(token)
  if updates and (step<=updates[-1] or (not sparse and step!=updates[-1]+1)):
   raise ValueError(f'{path}:{line}: duplicate, unordered or missing optimizer update ({updates[-1]} -> {step})')
  updates.append(step)
  for key in LOSSES:
   try:value=float(row[key])
   except (TypeError,ValueError) as error:raise ValueError(f'{path}:{line}: invalid {key}') from error
   if not math.isfinite(value):raise ValueError(f'{path}:{line}: nonfinite {key}')
   values[key].append(value)
 if not sparse and updates[0]!=1:raise ValueError(f'{path}: complete training history must start at optimizer_update1')
 return {'path':path,'raw':raw,'sha256':sha(raw),'fields':fields,'rows':rows,'updates':updates,'values':values,'sparse':sparse}


def ewma(values,span):
 alpha=2./(span+1);result=[];state=None
 for value in values:
  state=value if state is None else alpha*value+(1-alpha)*state
  result.append(state)
 return result


def write_smoothed(path,snapshot,spans):
 fields=list(snapshot['fields']);smoothed={}
 for loss in LOSSES:
  for span in spans:
   name=f'{loss}_ewma{span}'
   if name in fields:raise ValueError('Input already contains a requested derived EWMA column: '+name)
   fields.append(name);smoothed[name]=ewma(snapshot['values'][loss],span)
 with path.open('w',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
  for index,row in enumerate(snapshot['rows']):writer.writerow({**row,**{key:values[index] for key,values in smoothed.items()}})
 return smoothed


def render(metrics,output_dir,*,validation=None,spans=(100,500)):
 """Snapshot validation precedes any output mutation; returned SHA binds inputs."""
 if not spans or len(set(spans))!=len(spans) or any(type(span) is not int or span<1 for span in spans):
  raise ValueError('EWMA spans must be distinct positive integers')
 train=read_snapshot(metrics);val=read_snapshot(validation,sparse=True) if validation else None
 if val and not set(val['updates'])<=set(train['updates']):raise ValueError('Validation updates must correspond to recorded training rows')
 output=Path(output_dir)
 if val is None and any((output/name).exists() for name in ['validation_raw_snapshot.csv','validation_ewma.csv']):
  raise ValueError('Output contains previous validation artifacts; supply --validation or use a new output directory')
 names={'metrics_raw_snapshot.csv','metrics_ewma.csv','validation_raw_snapshot.csv','validation_ewma.csv','loss_curves.png','loss_curves.svg','plot_receipt.json'}
 for item in [train,val]:
  if item is not None and any(item['path'].resolve()==(output/name).resolve() for name in names):
   raise ValueError('Output artifacts must not replace an input CSV')
 import matplotlib
 matplotlib.use('Agg',force=True)
 from matplotlib import pyplot as plt
 output.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='.plot-staging-',dir=output) as directory:
  stage=Path(directory);(stage/'metrics_raw_snapshot.csv').write_bytes(train['raw'])
  smooth=write_smoothed(stage/'metrics_ewma.csv',train,spans)
  if val:
   (stage/'validation_raw_snapshot.csv').write_bytes(val['raw']);write_smoothed(stage/'validation_ewma.csv',val,spans)
  fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True,constrained_layout=True)
  try:
   for axis,loss in zip(axes,LOSSES):
    axis.plot(train['updates'],train['values'][loss],color='0.55',alpha=.4,linewidth=.7,label='Training raw')
    for span in spans:axis.plot(train['updates'],smooth[f'{loss}_ewma{span}'],linewidth=1.2,label=f'Training EWMA {span}')
    if val:axis.scatter(val['updates'],val['values'][loss],color='tab:red',s=12,label='Validation observed')
    axis.set_ylabel(loss);axis.grid(alpha=.2);axis.legend(loc='best',fontsize=9)
   axes[-1].set_xlabel('Optimizer update')
   axes[0].set_title('Q/V losses (all recorded optimizer updates)')
   fig.savefig(stage/'loss_curves.png',dpi=160);fig.savefig(stage/'loss_curves.svg')
  finally:plt.close(fig)
  report={'schema_version':1,'status':'RENDERED_COMPLETE_RECORDED_HISTORY','observed_at':datetime.now(timezone.utc).isoformat(),
   'training':{'input_path':str(train['path'].resolve()),'snapshot_sha256':train['sha256'],'rows':len(train['rows']),
    'first_optimizer_update':train['updates'][0],'last_optimizer_update':train['updates'][-1],
    'input_changed_after_snapshot':sha(train['path'].read_bytes())!=train['sha256']},
   'validation':None if val is None else {'input_path':str(val['path'].resolve()),'snapshot_sha256':val['sha256'],'observed_rows':len(val['rows']),
    'input_changed_after_snapshot':sha(val['path'].read_bytes())!=val['sha256'],'sparse':True,'interpolated':False},
   'ewma':{'spans':list(spans),'alpha':'2/(span+1)','initial_value':'first observed loss','bias_adjustment':False,
    'unit':'observed rows of each CSV; sparse validation gaps are not filled'},
   'invalid_input_policy':'Reject nonfinite losses, partial rows, duplicate/unordered updates, or missing training updates.',
   'source_inputs_modified':False,'renderer':'matplotlib Agg','matplotlib_version':matplotlib.__version__,
   'script_sha256':sha(Path(__file__).read_bytes()),
   'artifacts':{file.name:{'sha256':sha(file.read_bytes()),'bytes':file.stat().st_size} for file in sorted(stage.iterdir())}}
  for file in sorted(stage.iterdir()):os.replace(file,output/file.name)
  temporary=output/f'.plot_receipt.{os.getpid()}.tmp';temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');os.replace(temporary,output/'plot_receipt.json')
 return report


def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('metrics',type=Path)
 parser.add_argument('--validation',type=Path,help='Optional sparse validation.csv; no interpolation')
 parser.add_argument('--output-dir',type=Path);parser.add_argument('--spans',type=int,nargs='+',default=[100,500])
 args=parser.parse_args(argv)
 result=render(args.metrics,args.output_dir or args.metrics.parent/'loss_plots',validation=args.validation,spans=tuple(args.spans))
 print(json.dumps({'status':result['status'],'training':result['training'],'validation':result['validation']}))

if __name__=='__main__':main()
