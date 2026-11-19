"""CPU regression checks for dtype, exceptional steps, and durable journal recovery."""
import copy
import json
import tempfile
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from . import train
from .checkpoints import checkpoint, load_cp, restore, line, reconcile, retain
from .config import DEFAULT_CONFIG, QConfig, resolve_model_config
from .critic import Critic
from .data import save_json
from .train import Trainer


TEST_SEED = 37


class Samples:
    def __init__(self):
        self.calls=0
    def batch(self,rng,device):
        self.calls+=1
        self.last_draws=rng.integers(0,30,2)
        features=torch.ones(2,len(DEFAULT_CONFIG.camera_keys),4,16,dtype=torch.float16)
        actions=torch.ones(2,DEFAULT_CONFIG.horizon,DEFAULT_CONFIG.action_dim)
        mask=torch.ones(2,DEFAULT_CONFIG.horizon,dtype=torch.bool)
        return dict(features=features,next_features=features,action=actions,next_action=actions,mask=mask,next_mask=mask,
                    reward=torch.zeros(2),bootstrap=torch.ones(2,dtype=torch.bool))


def replay_check():
    ds=Samples()
    def small(layers,config):
        return Critic(layers=1,dim=16,heads=2,patches=4,config=config)
    with patch.object(train,'Critic',small):
        t=Trainer(torch.device('cpu'),1,TEST_SEED)
    seen=[]
    def check(_,args):
        assert args[0].dtype==torch.float32
        seen.append(str(args[0].dtype))
    t.net.register_forward_pre_hook(check)
    t.target.register_forward_pre_hook(check)
    dtype=[]
    def shared(_,args):
        dtype.append(args[2].dtype)
        if args[2].requires_grad:
            args[2].register_hook(lambda g: dtype.append(g.dtype))
    t.net.layers[0].register_forward_pre_hook(shared)
    events={}
    record=lambda event:events.update({event['step']:copy.deepcopy(event)})
    good=t.step(ds,record)
    assert good['optimizer_step'] and t.updates==1
    initial=[p.clone() for p in list(t.net.parameters())+list(t.target.parameters())]
    step1=[v['step'].clone() for v in t.optimizer.state.values()]
    hook_calls=[]
    def corrupt(g):
        hook_calls.append(1)
        return torch.full_like(g,float('nan')) if len(hook_calls)!=2 else g
    handle=t.net.head[-1].bias.register_hook(corrupt)
    skipped=t.step(ds,record)
    assert ds.calls==2 and skipped['outcome']=='SKIPPED' and not skipped['optimizer_step'] and not skipped['replay_nonfinite']
    e=events[2]
    assert e['first']['nonfinite_grad_params']==['head.4.bias'] and e['first']['logits']==e['replay']['logits'] and e['first']['loss']==e['replay']['loss']
    assert e['batch_indices']==ds.last_draws.tolist() and all(torch.equal(a,b) for a,b in zip(initial,list(t.net.parameters())+list(t.target.parameters())))
    assert all(torch.equal(v['step'],n) for v,n in zip(t.optimizer.state.values(),step1))
    second=t.step(ds,record)
    assert second['replay_nonfinite'] and second['stop_reason']=='consecutive skipped steps reached 2' and ds.calls==3
    handle.remove()
    resumed=t.step(ds,record)
    assert resumed['optimizer_step'] and t.consecutive==0 and resumed['lr']==3e-4*4/2000
    t.skipped=5
    handle=t.net.head[-1].bias.register_hook(lambda g:torch.full_like(g,float('inf')))
    sixth=t.step(ds,record)
    assert sixth['stop_reason']=='cumulative skipped steps exceeded 5' and sixth['cumulative_skips']==6
    handle.remove()
    t.validation(ds)
    assert seen and dtype and all(k==torch.float32 for k in dtype)
    return dict(same_batch=True,same_logits_and_loss_on_replay=True,no_optimizer_or_ema_on_skip=True,
                temporary_and_persistent_nonfinite=True,second_consecutive_stops=True,sixth_total_stops=True,
                float32_online_target_validation_and_context_gradient=True)


def gradient_guard_check():
    assert not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled() and torch.backends.cuda.math_sdp_enabled()
    ds=Samples()
    def small(layers,config):
        return Critic(layers=1,dim=16,heads=2,patches=4,config=config)
    with patch.object(train,'Critic',small):
        t=Trainer(torch.device('cpu'),1,TEST_SEED)
    events=[]
    h=t.net.head[-1].bias.register_hook(lambda g:torch.full_like(g,200.))
    row=t.step(ds,events.append)
    total=torch.nn.utils.get_total_norm([p.grad for p in t.net.parameters() if p.grad is not None])
    assert row['optimizer_step'] and row['clipped'] and row['grad_norm']>1000 and float(total)<=10.001 and t.high_streak==1
    t.high_streak=9
    done=t.updates
    row=t.step(ds,events.append)
    assert row['stop_reason'].startswith('gradient explosion') and not row['optimizer_step'] and t.updates==done
    h.remove(); t.high_streak=0
    h=t.net.head[-1].bias.register_hook(lambda g:torch.full_like(g,2e4))
    row=t.step(ds,events.append)
    assert row['stop_reason'].startswith('gradient explosion') and t.updates==done and len(events)==2
    h.remove()
    row=t.step(ds,events.append)
    assert row['optimizer_step'] and not row['clipped']
    return dict(math_attention=True,clip_to_10=True,sustained_high_stops=True,single_spike_stops=True)


def recovery_check():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        history=[dict(step=i) for i in range(1,5)]
        (root/'steps.jsonl').write_bytes(b''.join(line(r) for r in history)+b'{"step":5')
        reconcile(root,history)
        assert (root/'steps.jsonl').read_bytes()==b''.join(line(r) for r in history)
        assert next(root.glob('interruptions/*/previous_steps.jsonl')).read_bytes().endswith(b'{"step":5')
        for step in (0,2000,4000,6000,8000,10000,12000,14000,16000):
            p=root/f'step_{step:05d}.pt'
            p.touch()
            save_json(p.with_suffix('.pt.json'),dict(step=step,file=p.name))
            retain(root)
        assert sorted(p.name for p in root.glob('*.pt'))==['step_10000.pt','step_14000.pt','step_16000.pt']
        folder=root/f'seed_{TEST_SEED}'
        folder.mkdir()
        def small(layers,config):
            return Critic(layers=1,dim=16,heads=2,patches=4,config=config)
        with patch.object(train,'Critic',small):
            t=Trainer(torch.device('cpu'),1,TEST_SEED)
            ds=Samples()
            row=t.step(ds,lambda event:None)
            checkpoint(folder,t,[row],[],TEST_SEED,'synthetic-protocol','synthetic-cache')
            second=t.step(ds,lambda event:None)
            draws=ds.last_draws.copy()
            expected=copy.deepcopy(t.net.state_dict())
            new=Trainer(torch.device('cpu'),1,TEST_SEED)
            h,_=restore(new,load_cp(folder,1,'synthetic-protocol','synthetic-cache'))
            assert h==[row] and new.updates==1
            actual=new.step(ds,lambda event:None)
            assert actual==second and np.array_equal(ds.last_draws,draws)
            assert all(torch.equal(v,expected[n]) for n,v in new.net.state_dict().items())
    return dict(rolling_retention=True,checkpoint_prefix_and_partial_journal_recovery=True,
                cpu_resume_rng_optimizer_and_next_step_exact=True)


def entrypoint_check(config):
    """Use short synthetic roots, a mock DINO and a tiny Q through the normal fit CLI."""
    from . import features, fit, c_score
    from .data import sha

    class Dino(torch.nn.Module):
        def forward(self,pixel_values):
            values=pixel_values.mean((1,2,3))[:,None,None]
            return SimpleNamespace(last_hidden_state=values.expand(-1,257,1024))

    class ToyQ(Critic):
        def __init__(self,layers=1,config=DEFAULT_CONFIG):
            super().__init__(layers=1,dim=16,heads=2,patches=4,config=config)
        def forward(self,features,actions,mask):
            return super().forward(features[:,:,:4,:16],actions,mask)

    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        dino=root/'dino'
        dino.mkdir()
        hashes={}
        for name in features.DINO_FILES:
            path=dino/name
            path.write_bytes(b'synthetic model file')
            hashes[name]=sha(path)
        save_json(root/'qconfig.json',config.to_dict())
        save_json(root/'dino_receipt.json',dict(sha256=hashes))
        rows=[]
        for index,(split,n,success) in enumerate((('train',9,False),('train',6,True),('val',5,False),('val',8,True))):
            rid=f'r{index}'
            ep=root/rid/'episodes/episode_000'
            ep.mkdir(parents=True)
            actions=np.arange(n*config.action_dim,dtype=np.float32).reshape(n,config.action_dim)/100
            trajectory=ep/'trajectory.npz'
            np.savez(trajectory,env_action=actions)
            images=np.zeros((n+1,16,16,3),dtype=np.uint8)
            observations=ep/'trajectory_observations.npz'
            np.savez(observations,**{key:images+i*70 for i,key in enumerate(config.camera_keys)})
            rows.append(dict(run_id=rid,root_run_id=rid,kind='original',split=split,label_success=success,
                             platform_dir=rid,terminal_step=n-2 if success else None,
                             trajectory_sha256=sha(trajectory),observations_sha256=sha(observations)))
        save_json(root/'manifest.json',dict(members=rows))
        cache=root/'features'
        output=root/'training'
        args=['fit','--manifest',str(root/'manifest.json'),'--dino-dir',str(dino),'--cache',str(cache),
              '--output',str(output),'--seed',str(TEST_SEED),'--device','cpu',
              '--config',str(root/'qconfig.json'),'--dino-receipt',str(root/'dino_receipt.json')]
        with (patch.object(sys,'argv',args),
              patch.object(features.Dinov2Model,'from_pretrained',return_value=Dino()),
              patch.object(train,'Critic',ToyQ), patch.object(fit,'STEPS',2)):
            fit.main()
            idx=json.loads((cache/'index.json').read_text())
            assert [len(r['frames']) for r in idx['roots']]==[min(n,config.max_steps) for n in (9,5,5,7)]
            folder=output/f'seed_{TEST_SEED}'
            result=json.loads((folder/'complete.json').read_text())
            assert result['step']==2 and set(r['id'] for r in result['validation']['roots'])=={'r2','r3'}
            assert (folder/result['model_file']).is_file()
            stats_path=folder/'train_action_stats.json'
            assert json.loads(stats_path.read_text())['n']==min(9,config.max_steps)+min(5,config.max_steps)
            cached=np.load(cache/'r0.npy',mmap_mode='r')
            assert cached.shape[1:]==(len(config.camera_keys),256,1024)
            assert np.all(np.diff(cached[0,:,0,0])>0) # camera identity and order preserved
            candidates=root/'candidate.npz'
            np.savez(candidates,features=cached[:1],actions=np.zeros((1,2,config.horizon,config.action_dim),np.float32))
            exported=folder/result['model_file']
            payload=torch.load(exported,map_location='cpu',weights_only=True)
            assert resolve_model_config(payload)==config
            weights_only=root/'legacy.pt'
            torch.save(payload['model'],weights_only)
            predictions=[]
            for model_path,options in ((exported,[]),(weights_only,['--config',str(root/'qconfig.json')])):
                out=root/'scores.json'
                argv=['score','--checkpoint',str(model_path),'--candidates',str(candidates),
                      '--action-stats',str(stats_path),'--output',str(out),*options]
                with patch.object(sys,'argv',argv),patch.object(c_score,'Critic',ToyQ):
                    c_score.main()
                predictions.append(json.loads(out.read_text())['scores'])
            np.testing.assert_array_equal(*predictions)
            assert np.asarray(predictions[0]).shape==(1,2)
            if config != DEFAULT_CONFIG:
                try:
                    resolve_model_config(payload,DEFAULT_CONFIG)
                except ValueError:
                    pass
                else:
                    raise AssertionError('explicit Q layout mismatch accepted')
                assert not result['validation']['roots'][-1]['success']  # success occurs beyond the shortened episode
            journal=(folder/'steps.jsonl').read_bytes()
            assert len(journal.splitlines())==2 and all(json.loads(r)['optimizer_step'] for r in journal.splitlines())
            fit.main()   # re-audit the existing cache, retain the completed journal
            save_json(folder/'stopped.json',dict(stop_reason='nonfinite parameters before forward'))
            try:
                fit.main()
            except RuntimeError as error:
                assert 'recorded training stop' in str(error)
            else:
                raise AssertionError('terminal seed was restarted')
            assert (folder/'steps.jsonl').read_bytes()==journal
    return dict(cpu_cli=True,variable_length_train_val_windows=True,atomic_cache_build_and_reaudit=True,
                validation_and_export=True,terminal_stop_without_event_is_durable=True,shared_config=config.to_dict(),
                bare_and_configured_model_scores_match=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    print(json.dumps(dict(passed=True,training=replay_check(),gradient_guard=gradient_guard_check(),recovery=recovery_check(),entrypoint=entrypoint_check(DEFAULT_CONFIG),
                          alternate=entrypoint_check(QConfig(6,3,4,('side','top','zoom'),
                              ('env_side','env_top','env_zoom'),('actor_side','actor_top','actor_zoom')))),indent=2))
