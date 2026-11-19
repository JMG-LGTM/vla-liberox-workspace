"""Paired action-ranking metrics and state-level resampling."""
import itertools

import numpy as np

BOOT = 5000


def cv_oracle(ok):
    out = []
    for a, b in itertools.permutations(range(ok.shape[1]), 2):
        best = np.flatnonzero(ok[:, a] == ok[:, a].max())
        out.append(ok[best, b].mean())
    return float(np.mean(out))


def pool_metrics(q, ok, rate):
    best = np.flatnonzero(q == q.max())
    pairs = [(i, j) for i, j in itertools.combinations(range(len(q)), 2) if rate[i] != rate[j]]
    acc = [1.0 if (q[i] - q[j]) * (rate[i] - rate[j]) > 0 else 0.5 if q[i] == q[j] else 0.0 for i, j in pairs]
    return rate[best].mean(), rate.mean(), cv_oracle(ok), acc


def summarize(rows, rng):
    pick, rand, orc = (np.array([r[i] for r in rows]) for i in range(3))
    d = pick - rand
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    acc = [a for r in rows for a in r[3]]
    gain = orc.mean() - rand.mean()
    return dict(pick=pick.mean(), random=rand.mean(), oracle_cv=orc.mean(), q_minus_random=d.mean(),
                ci95=np.percentile(boot, [2.5, 97.5]).tolist(), share_of_oracle_gain=d.mean() / gain if gain > 0.03 else None,
                pairwise_accuracy=float(np.mean(acc)) if acc else None, pairs=len(acc))


def rows_for(scores, labels):
    return [pool_metrics(q,ok,ok.mean(1)) for q,ok in zip(scores,labels)]


def average_metrics(scores,labels,*,seed=0):
    # Average independent policies' metrics within each state before the paired bootstrap.
    sets=[rows_for(q,labels) for q in scores]
    states=[]
    for group in zip(*sets):
        states.append((*np.mean([row[:3] for row in group],axis=0),np.mean([row[3] for row in group],axis=0).tolist()))
    return summarize(states,np.random.default_rng(seed))


def permutation(scores,labels,*,seed=0):
    rate=labels.mean(2)
    best=scores==scores.max(1,keepdims=True)
    weights=best/best.sum(1,keepdims=True)
    observed=float((weights*rate).sum(1).mean()-rate.mean())
    rng=np.random.default_rng(seed)
    indices=rng.random((BOOT,*rate.shape)).argsort(2)
    perm=np.take_along_axis(rate[None,:,:],indices,axis=2)
    null=(weights[None,:,:]*perm).sum(2).mean(1)-rate.mean()
    return dict(statistic=observed,permutations=BOOT,mean=float(null.mean()),ci95=np.percentile(null,[2.5,97.5]).tolist(),
                p_one_sided=float((1+np.count_nonzero(null>=observed-1e-12))/(1+BOOT)))


def baseline(labels,*,seed=0):
    labels = np.asarray(labels, dtype=np.float64)
    rand=average_metrics([np.zeros(labels.shape[:2])],labels,seed=seed)
    rand['pairwise_accuracy']=.5
    # A ranking learned on one repeat, measured on the other; average both directions.
    dirs=[]
    for a,b in ((0,1),(1,0)):
        held=np.repeat(labels[:,:,b:b+1],2,axis=2)
        dirs.append(rows_for(labels[:,:,a],held))
    rows=[]
    for left,right in zip(*dirs):
        rows.append((*((np.asarray(left[:3])+right[:3])/2),left[3]+right[3]))
    oracle=summarize(rows,np.random.default_rng(seed))
    assert np.isclose(oracle['pick'],rand['oracle_cv'])
    return dict(random=rand,oracle_cv=oracle)
