'Evaluate approximate updates against the retraining reference.'
import numpy as np
import os
from .cli import parser, load, make_backend, split
from . import core
from . import unlearning as U


def main() -> int:
    p = parser(__doc__)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = load(a)
    (be, X) = make_backend(a, c)
    y = c['y']
    (tr, te) = split(len(y), 0)
    full = be.fit(X[tr], y[tr], seed=0)
    test_mse = float(np.mean((full.predict(X[te]) - y[te]) ** 2))
    rng = np.random.default_rng(1)
    reqs = rng.choice(tr, size=min(a.n_requests, len(tr)), replace=False)
    print(f'measuring {len(reqs)} floors x {a.seeds} seeds ({len(reqs) * a.seeds} retrains, shared across all {len(U.METHODS)} methods)')
    floors = {}
    for i in reqs:
        floors[int(i)] = float(np.median([core.measure_floor(be, X, y, [i], i, seed=s, train_idx=tr) for s in range(a.seeds)]))
    rows = []
    (Xtr, ytr) = (X[tr], y[tr])
    pos = {int(g): k for (k, g) in enumerate(tr)}
    for (name, fn) in U.METHODS.items():
        (a1, a2, td) = ([], [], [])
        for i in reqs:
            floor = floors[int(i)]
            m = fn(be, Xtr, ytr, [pos[int(i)]], seed=0, original=full)
            err = float((float(m.predict(X[i][None, :])[0]) - float(y[i])) ** 2)
            a1.append(abs(err - floor))
            a2.append(err / max(floor, 1e-12))
            td.append(float(np.mean((m.predict(X[te]) - y[te]) ** 2)) / max(test_mse, 1e-12))
        (a1_lo, a1_hi) = core.bootstrap_ci(np.asarray(a1))
        (a2_lo, a2_hi) = core.bootstrap_ci(np.asarray(a2))
        rows.append((name, float(np.median(a1)), a1_lo, a1_hi, float(np.median(a2)), a2_lo, a2_hi, float(np.percentile(a2, 25)), float(np.percentile(a2, 75)), float(np.median(td))))
        print(f'{name:>28} gap {np.median(a1):.3e} [{a1_lo:.2e}, {a1_hi:.2e}]  err/floor {np.median(a2):.3f} [{a2_lo:.3f}, {a2_hi:.3f}]  test {np.median(td):.3f}')
    core.write_csv(os.path.join(a.out, 'summary.csv'), ['method', 'axis1_gap', 'axis1_ci_lo', 'axis1_ci_hi', 'axis2_err_over_floor', 'axis2_ci_lo', 'axis2_ci_hi', 'q25', 'q75', 'test_mse_ratio'], rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
