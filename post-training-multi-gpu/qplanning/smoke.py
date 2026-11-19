"""Run 5 warmup and 24 measured updates on the configured frozen features."""
import argparse
from pathlib import Path
import time

import torch
from .config import load_config
from .data import save_json
from .features import Samples
from .fit import platform_lock, prepare
from .train import Trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','dino-dir','cache','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--dino-receipt',type=Path)
    parser.add_argument('--validation-seed',type=int,default=0)
    parser.add_argument('--cache-receipt',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    parser.add_argument('--gpu-lock',type=Path)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    with platform_lock(device,args.gpu_lock):
        config=load_config(args.config)
        rows,stats,_,_=prepare(args,device,config)
        train=Samples(args.cache,'train',rows,stats,config)
        val=Samples(args.cache,'val',rows,stats,config)
        trainer=Trainer(device,18,args.seed,config,args.validation_seed)
        before=trainer.net.head[-1].weight.detach().clone()
        def event(record):
            save_json(args.output/f"smoke_event_{record['step']:05d}.json",record)
        measured=[]
        for t in range(1,30):
            if t==6:
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                begin=time.monotonic()
            row=trainer.step(train,event)
            if row.get('stop_reason'):
                raise FloatingPointError(row['stop_reason'])
            if t>5:
                measured.append(row)
        if device.type=='cuda':
            torch.cuda.synchronize(device)
        elapsed=time.monotonic()-begin
        shifts=[float((net.head[-1].weight.detach()-before).abs().max()) for net in (trainer.net,trainer.target)]
        if min(shifts)<=0 or not all(torch.isfinite(net.head[-1].weight).all() for net in (trainer.net,trainer.target)):
            raise ValueError('smoke did not produce finite online and target updates')
        save_json(args.output/'smoke.json',dict(seed=args.seed,layers=18,measured=measured,device=str(device),
                 updates_per_second=24/elapsed,parameter_shifts=shifts,validation=trainer.validation(val)))


if __name__=='__main__':
    main()
