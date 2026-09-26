"""Measure retraining loss across controlled redundancy settings."""
import numpy as np
RNG = np.random.default_rng(20260814)
(DIM, NFEAT, SIGMA_W, NOISE, LAM) = (6, 300, 0.05, 0.02, 0.0001)

def rff(X, W, b):
    return np.sqrt(2.0 / W.shape[0]) * np.cos(np.atleast_2d(X) @ W.T + b)

def make_world(n_families=140, per_family=4, n_isolated=160):
    W = RNG.normal(size=(NFEAT, DIM))
    b = RNG.uniform(0, 2 * np.pi, size=NFEAT)
    truth = RNG.normal(size=NFEAT) * 0.3
    f = lambda X: rff(X, W, b) @ truth
    return (W, b, f)

def build_corpus(f, n_families, per_family, n_isolated, sigma_w=SIGMA_W):
    centers = RNG.normal(size=(n_families, DIM)) * 1.5
    fam = np.repeat(centers, per_family, axis=0) + RNG.normal(size=(n_families * per_family, DIM)) * sigma_w
    iso = RNG.normal(size=(n_isolated, DIM)) * 1.5
    X = np.vstack([fam, iso])
    fam_id = np.concatenate([np.repeat(np.arange(n_families), per_family), -np.ones(n_isolated, dtype=int)])
    y = f(X) + RNG.normal(size=len(X)) * NOISE
    return (X, y, fam_id)

def fit(Phi, y):
    A = Phi.T @ Phi + LAM * len(Phi) * np.eye(Phi.shape[1])
    return np.linalg.solve(A, Phi.T @ y)

def sq(a, b):
    return float((a - b) ** 2)

def redundancy_sweep(reps=300):
    rows = []
    (W, b, f) = make_world()
    Xte = RNG.normal(size=(1500, DIM)) * 1.5
    yte = f(Xte)
    Pte = rff(Xte, W, b)
    for k in [1, 2, 4, 8, 16]:
        (floors, isofloors) = ([], [])
        for _ in range(reps):
            (X, y, fam) = build_corpus(f, 90, k, 160)
            Phi = rff(X, W, b)
            w_full = fit(Phi, y)
            test_mse = float(np.mean((Pte @ w_full - yte) ** 2))
            cand = np.flatnonzero(fam >= 0)
            i = int(RNG.choice(cand))
            keep = np.ones(len(X), bool)
            keep[i] = False
            w_r = fit(Phi[keep], y[keep])
            floors.append(sq(Phi[i] @ w_r, y[i]) / max(test_mse, 1e-12))
            cand2 = np.flatnonzero(fam < 0)
            j = int(RNG.choice(cand2))
            keep2 = np.ones(len(X), bool)
            keep2[j] = False
            w_r2 = fit(Phi[keep2], y[keep2])
            isofloors.append(sq(Phi[j] @ w_r2, y[j]) / max(test_mse, 1e-12))
        fl = np.asarray(floors)
        iso = np.asarray(isofloors)
        bs = np.array([np.median(RNG.choice(fl, fl.size, replace=True)) for _ in range(2000)])
        bsi = np.array([np.median(RNG.choice(iso, iso.size, replace=True)) for _ in range(2000)])
        rows.append((k, float(np.median(fl)), float(np.percentile(fl, 25)), float(np.percentile(fl, 75)), float(np.median(iso)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(np.percentile(bsi, 2.5)), float(np.percentile(bsi, 97.5))))
    return rows

def main() -> int:
    import argparse
    import csv
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Controlled redundancy study')
    parser.add_argument('--out', type=Path, default=Path('development/runs/controlled'))
    parser.add_argument('--reps', type=int, default=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'floor_vs_redundancy.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['k', 'floor_median', 'floor_q25', 'floor_q75', 'isolated_median', 'floor_ci_low', 'floor_ci_high', 'isolated_ci_low', 'isolated_ci_high'])
        writer.writerows(redundancy_sweep(args.reps))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
