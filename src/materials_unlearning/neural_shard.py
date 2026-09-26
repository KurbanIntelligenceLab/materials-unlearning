"""Compute a shard of neural deletion requests."""
import argparse
import csv
import os
import sys
import time
import numpy as np
from . import core
from .cli import load, make_backend, parser, split

def done_ids(path):
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {row['id'] for row in csv.DictReader(fh) if row.get('id')}

def main():
    p = parser(__doc__)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--n-shards', type=int, default=1)
    p.add_argument('--loo-control', action='store_true', help='also measure the leave-one-out control arm for each request')
    a = p.parse_args()
    if a.n_shards < 1 or not 0 <= a.shard < a.n_shards:
        p.error("require --n-shards >= 1 and 0 <= --shard < --n-shards")
    os.makedirs(a.out, exist_ok=True)
    c = load(a)
    (be, X) = make_backend(a, c)
    y = c['y']
    (tr, te) = split(len(y), 0)
    t0 = time.time()
    full = be.fit(X[tr], y[tr], seed=0)
    fit_s = time.time() - t0
    test_mse = float(np.mean((full.predict(X[te]) - y[te]) ** 2))
    r2 = 1.0 - test_mse / float(np.var(y[te]))
    L_hat = core.empirical_lipschitz(full.predict, X[tr])
    eps_bar = core.training_error_level(full.predict, X[tr], y[tr])
    print(f'corpus {len(y)} | train {len(tr)} | backend {a.backend} | one fit {fit_s / 60:.1f} min', flush=True)
    print(f'test MSE {test_mse:.5f} | R2 {r2:.3f} | L_hat {L_hat:.4f} | eps_bar {eps_bar:.4f}', flush=True)
    if r2 < 0.5:
        print('WARNING: R2 below 0.5; weak model fit limits interpretation of deletion losses.', flush=True)
    rng = np.random.default_rng(0)
    cand = rng.choice(tr, size=min(a.n_requests, len(tr)), replace=False)
    mine = cand[a.shard::a.n_shards]
    probe = np.empty(len(cand))
    for (k, i) in enumerate(cand):
        keep = np.setdiff1d(tr, [i])
        (probe[k], _, _) = core.floorprobe(X[i], y[i], X[keep], y[keep], L_hat, eps_bar)
    (strata, _) = core.stratify(probe)
    strat_of = {int(i): s for (i, s) in zip(cand, strata)}
    probe_of = {int(i): float(v) for (i, v) in zip(cand, probe)}
    path = os.path.join(a.out, f'neural_shard{a.shard}of{a.n_shards}.csv')
    already = done_ids(path)
    new = os.path.getsize(path) == 0 if os.path.exists(path) else True
    fh = open(path, 'a', newline='')
    w = csv.writer(fh)
    if new:
        w.writerow(['id', 'idx', 'stratum', 'probe', 'floor', 'floor_over_testmse', 'loo_floor', 'test_mse', 'r2', 'n_features', 'epochs', 'seconds'])
        fh.flush()
    todo = [i for i in mine if str(c['ids'][int(i)]) not in already]
    print(f'shard {a.shard}/{a.n_shards}: {len(mine)} requests, {len(mine) - len(todo)} already done, {len(todo)} to run (~{len(todo) * fit_s * (2 if a.loo_control else 1) / 3600:.1f} h)', flush=True)
    for (n, i) in enumerate(todo, 1):
        i = int(i)
        t = time.time()
        fl = float(np.median([core.measure_floor(be, X, y, [i], i, seed=s, train_idx=tr) for s in range(a.seeds)]))
        loo = ''
        if a.loo_control:
            j = int(rng.choice(np.setdiff1d(tr, [i])))
            loo = float(core.measure_displacement_control(be, X, y, [j], i, seed=0, train_idx=tr))
        dt = time.time() - t
        w.writerow([str(c['ids'][i]), i, strat_of[i], f'{probe_of[i]:.6g}', f'{fl:.6g}', f'{fl / max(test_mse, 1e-12):.6g}', f'{loo:.6g}' if loo != '' else '', f'{test_mse:.6g}', f'{r2:.4f}', a.n_features, a.epochs, f'{dt:.1f}'])
        fh.flush()
        os.fsync(fh.fileno())
        print(f"  [{n}/{len(todo)}] {c['ids'][i]} {strat_of[i]:>9} floor/testMSE {fl / max(test_mse, 1e-12):.4f}  ({dt / 60:.1f} min)", flush=True)
    fh.close()
    print(f'shard {a.shard} complete -> {path}', flush=True)
if __name__ == '__main__':
    main()
