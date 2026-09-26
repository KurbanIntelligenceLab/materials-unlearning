'Measure deletion losses and descriptor-based strata.'
import numpy as np
import os
import sys
from .cli import parser, load, make_backend, split
from . import core


def main() -> int:
    a = parser(__doc__).parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = load(a)
    (be, X) = make_backend(a, c)
    y = c['y']
    (tr, te) = split(len(y), 0)
    full = be.fit(X[tr], y[tr], seed=0)
    test_mse = float(np.mean((full.predict(X[te]) - y[te]) ** 2))
    L_hat = core.empirical_lipschitz(full.predict, X[tr])
    eps_bar = core.training_error_level(full.predict, X[tr], y[tr])
    print(f'test MSE {test_mse:.5f} | L_hat {L_hat:.4f} | eps_bar {eps_bar:.4f}')
    rng = np.random.default_rng(0)
    cand = rng.choice(tr, size=min(a.n_requests, len(tr)), replace=False)
    probe = []
    for i in cand:
        keep = np.setdiff1d(tr, [i])
        (p, _, _) = core.floorprobe(X[i], y[i], X[keep], y[keep], L_hat, eps_bar)
        probe.append(p)
    probe = np.asarray(probe)
    (strata, names) = core.stratify(probe)
    (rows, held) = ([], [])
    for (i, st, pv) in zip(cand, strata, probe):
        fl = [core.measure_floor(be, X, y, [i], i, seed=s, train_idx=tr) for s in range(a.seeds)]
        keep = np.setdiff1d(tr, [i])
        (_, j, _) = core.floorprobe(X[i], y[i], X[keep], y[keep], L_hat, eps_bar)
        m = be.fit(X[keep], y[keep], seed=0)
        held.append(abs(float(m.predict(X[keep][j][None, :])[0]) - float(y[keep][j])) <= eps_bar)
        rows.append((c['ids'][i], st, pv, float(np.median(fl)), float(np.median(fl)) / max(test_mse, 1e-12)))
    core.write_csv(os.path.join(a.out, 'deletion_losses.csv'), ['id', 'stratum', 'floorprobe', 'floor', 'floor_over_testmse'], rows)
    print(f'\nAssumption 1(ii) held in {100 * np.mean(held):.1f}% of requests')
    print(f"{'stratum':>12} {'n':>5} {'median floor/testMSE':>22} {'95% CI':>24}")
    summ = []
    for nm in names:
        v = np.array([r[4] for r in rows if r[1] == nm])
        if not len(v):
            continue
        (lo, hi) = core.bootstrap_ci(v)
        summ.append((nm, len(v), float(np.median(v)), lo, hi))
        print(f"{nm:>12} {len(v):>5} {np.median(v):>22.4f} {f'[{lo:.4f}, {hi:.4f}]':>24}")
    core.write_csv(os.path.join(a.out, 'floor_summary.csv'), ['stratum', 'n', 'median_floor_over_testmse', 'ci_lo', 'ci_hi'], summ)
    if len(summ) >= 2:
        r = summ[-1][2] / max(summ[0][2], 1e-12)
        print(f'\nisolated/redundant ratio: {r:.1f}x')
        print('Overlapping stratum confidence intervals do not establish separation.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
