"""One sampler draw and one scheduled step, including a skipped update."""
import copy
import math

import numpy as np
import torch
from .config import DEFAULT_CONFIG
from .critic import Critic, expectation, hl_gauss, td_target

GRAD_CLIP = 10.0      # same max norm as the Q-planning LeRobot training loop
GRAD_SPIKE = 1e5      # stop at once above this raw norm
GRAD_HIGH = 1e3       # stop after HIGH_STREAK consecutive raw norms above this
HIGH_STREAK = 10


def scalar(value):
    v = float(value)
    return v if math.isfinite(v) else ('NaN' if math.isnan(v) else ('+Inf' if v > 0 else '-Inf'))


def tensor_stats(x):
    x = x.detach().float()
    finite = x[torch.isfinite(x)]
    return dict(shape=list(x.shape), nan=int(x.isnan().sum()), posinf=int(x.isposinf().sum()),
                neginf=int(x.isneginf().sum()), min=float(finite.min()) if finite.numel() else None,
                max=float(finite.max()) if finite.numel() else None,
                mean=float(finite.double().mean()) if finite.numel() else None)


@torch.no_grad()
def param_max(net):
    return float(torch.stack(torch._foreach_norm(list(net.parameters()), float('inf'))).amax())


def torch_states(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def set_states(device, states):
    torch.set_rng_state(states[0])
    if states[1] is not None:
        torch.cuda.set_rng_state(states[1], device)


class Trainer:
    def __init__(self, device, layers, seed, config=DEFAULT_CONFIG, validation_seed=0):
        self.config = config
        self.validation_seed = validation_seed
        # Keep the training backend stable. Importing the critic for inference must not alter an actor's backend.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        self.device = device
        torch.manual_seed(seed)
        self.net = Critic(layers=layers,config=config).to(device)
        self.target = copy.deepcopy(self.net).requires_grad_(False).eval()
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=3e-4, weight_decay=1e-4, fused=device.type=='cuda')
        self.rng = np.random.default_rng(seed)
        self.updates = 0  # completed scheduled steps, including skips
        self.skipped = 0
        self.consecutive = 0
        self.high_streak = 0
        self.last_bounds = None
        self.maxima = None

    def objective(self, batch):
        mask = batch['bootstrap']
        self.outputs = {}
        with torch.no_grad(), torch.autocast(self.device.type, dtype=torch.bfloat16):
            following = torch.zeros_like(batch['reward'])
            if len(batch['next_features']):
                target = self.target(batch['next_features'].float(),batch['next_action'][mask],batch['next_mask'][mask])
                self.outputs['target'] = target.detach()
                following[mask] = expectation(target)
            y = td_target(batch['reward'],mask,following)
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            logits = self.net(batch['features'].float(),batch['action'],batch['mask'])
        self.outputs['online'] = logits.detach()
        q = expectation(logits.detach())
        ce = -(hl_gauss(y)*logits.float().log_softmax(-1)).sum(-1)
        self.outputs['per_example_loss'] = ce.detach()
        self.last_bounds = [float(q.min()),float(q.max())]
        return ce.mean()

    def backward(self, batch):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.objective(batch)
        norm = None
        if torch.isfinite(loss) and self.bounds_ok():
            loss.backward()
            grads = [p.grad for p in self.net.parameters() if p.grad is not None]
            # Same global L2 reduction as the original guard. No gradient scaling.
            norm = float(torch.nn.utils.get_total_norm(grads,2.0,error_if_nonfinite=False))
        return float(loss.detach()), norm

    def bounds_ok(self):
        return self.last_bounds[0] >= -.010001 and self.last_bounds[1] <= 1.010001

    def evidence(self,loss,norm):
        return dict(loss=scalar(loss), grad_norm=None if norm is None else scalar(norm),
                    logits={k:tensor_stats(x) for k,x in self.outputs.items() if k!='per_example_loss'},
                    per_example_loss=tensor_stats(self.outputs['per_example_loss']),
                    nonfinite_grad_params=[n for n,p in self.net.named_parameters()
                                           if p.grad is not None and not torch.isfinite(p.grad).all()])

    def step(self,samples,record_event):
        step = self.updates + 1
        lr = 3e-4*step/2000 if step<=2000 else 1e-6 + (3e-4-1e-6)*(.5+.5*math.cos(math.pi*(step-2000)/38000))
        self.optimizer.param_groups[0]['lr'] = lr
        if self.maxima is None:
            self.maxima = (param_max(self.net),param_max(self.target))
        before,target_before = self.maxima
        row = dict(step=step,outcome='STOPPED',optimizer_step=False,loss=None,grad_norm=None,logits_absmax=None,
                   param_absmax=scalar(before),param_absmax_after=scalar(before),lr=lr)
        if not all(math.isfinite(v) for v in self.maxima):
            row['stop_reason'] = 'nonfinite online or EMA parameters before forward'
            return row
        start_rng = torch_states(self.device)
        batch = samples.batch(self.rng,self.device)
        loss,norm = self.backward(batch)
        row.update(loss=scalar(loss),grad_norm=None if norm is None else scalar(norm),
                   logits_absmax=scalar(self.outputs['online'].abs().amax()),q_bounds=[scalar(v) for v in self.last_bounds])
        stop = None
        if norm is None:
            event=dict(step=step,batch_indices=samples.last_draws.tolist(),first=self.evidence(loss,norm))
            stop='nonfinite loss' if not math.isfinite(loss) else 'Q outside allowed range'
            record_event(dict(**event,stop_reason=stop))
        elif not math.isfinite(norm):
            event=dict(step=step,batch_indices=samples.last_draws.tolist(),first=self.evidence(loss,norm))
            record_event(event)
            after_first = torch_states(self.device)
            set_states(self.device,start_rng)
            try:
                loss2,norm2 = self.backward(batch)
                event['replay'] = self.evidence(loss2,norm2)
                event['replay']['nonfinite_grad_norm'] = None if norm2 is None else not math.isfinite(norm2)
            finally:
                set_states(self.device,after_first)
            self.optimizer.zero_grad(set_to_none=True)
            self.skipped += 1
            self.consecutive += 1
            self.updates = step
            row.update(outcome='SKIPPED',replay_nonfinite=event['replay']['nonfinite_grad_norm'])
            if norm2 is None:
                stop = 'nonfinite loss on replay' if not math.isfinite(loss2) else 'Q outside allowed range on replay'
            elif self.consecutive>=2 or self.skipped>5:
                stop = 'consecutive skipped steps reached 2' if self.consecutive>=2 else 'cumulative skipped steps exceeded 5'
            record_event(dict(**event,cumulative_skips=self.skipped,consecutive_skips=self.consecutive,stop_reason=stop))
        elif norm>GRAD_SPIKE or (norm>GRAD_HIGH and self.high_streak+1>=HIGH_STREAK):
            stop = f'gradient explosion: raw norm {norm:.4g}, high streak {self.high_streak+1}'
            record_event(dict(step=step,batch_indices=samples.last_draws.tolist(),first=self.evidence(loss,norm),stop_reason=stop))
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.high_streak = self.high_streak+1 if norm>GRAD_HIGH else 0
            torch.nn.utils.clip_grad_norm_(self.net.parameters(),GRAD_CLIP)
            self.optimizer.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(self.target.parameters()),list(self.net.parameters()),.005)
            self.maxima = (param_max(self.net),param_max(self.target))
            self.updates = step
            self.consecutive = 0
            row.update(outcome='UPDATED',optimizer_step=True,clipped=norm>GRAD_CLIP,param_absmax_after=scalar(self.maxima[0]))
            if not all(math.isfinite(v) for v in self.maxima):
                stop = 'nonfinite online or EMA parameters after update'
                record_event(dict(step=step,batch_indices=samples.last_draws.tolist(),stop_reason=stop,
                                  max_online=scalar(self.maxima[0]),max_target=scalar(self.maxima[1])))
        if stop:
            row['stop_reason'] = stop
        row.update(cumulative_skips=self.skipped,consecutive_skips=self.consecutive)
        return row

    @torch.no_grad()
    def validation(self,samples):
        self.net.eval()
        try:
            loss=self.objective(samples.batch(np.random.default_rng(self.validation_seed),self.device))
            if not torch.isfinite(loss) or not self.bounds_ok():
                raise FloatingPointError('validation loss or Q outside allowed range')
            return dict(loss=float(loss),q_bounds=[scalar(v) for v in self.last_bounds])
        finally:
            self.net.train()
