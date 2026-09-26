'Compute exact ridge deletion losses and request-level diagnostics.'
from pathlib import Path
from .configuration import RepositoryPaths, RidgeConfiguration
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def main() -> int:
    layout = RepositoryPaths()
    REPO = layout.root
    START = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=int, default=2048)
    parser.add_argument('--lam', type=float, default=1e-06)
    parser.add_argument('--corpus-size', type=int, default=0)
    parser.add_argument('--diagnostic-requests', type=int, default=200)
    parser.add_argument('--out', type=Path, default=Path('development/runs/ridge'))
    args = parser.parse_args()
    try:
        config = RidgeConfiguration(args.features, args.lam, args.diagnostic_requests, args.corpus_size)
    except ValueError as exc:
        parser.error(str(exc))
    (NF, LAM) = (config.features, config.regularization)
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    def write_csv(name: str, rows: list[dict[str, str | int | float]]):
        with (OUT / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with np.load(layout.descriptor_cache, allow_pickle=False) as cache:
        raw = cache['X'].astype(float)
        y = cache['y'].astype(float)
        ids = cache['ids'].astype(str)
    if config.corpus_size:
        (raw, y, ids) = (raw[:config.corpus_size], y[:config.corpus_size], ids[:config.corpus_size])
    perm = np.random.default_rng(0).permutation(len(y))
    (te, tr) = (perm[:int(0.2 * len(y))], perm[int(0.2 * len(y)):])
    if len(tr) < max(config.diagnostic_requests, 200):
        parser.error('the training partition must contain at least 200 rows and cover all diagnostic requests')
    (mu, sd) = (raw[tr].mean(0), raw[tr].std(0) + 1e-09)
    x = (raw - mu) / sd
    pair_rng = np.random.default_rng(0)
    p1 = pair_rng.integers(len(tr), size=4000)
    p2 = pair_rng.integers(len(tr), size=4000)
    distances = np.linalg.norm(x[tr[p1]] - x[tr[p2]], axis=1)
    gamma = 1 / np.median(distances[distances > 0])
    basis_rng = np.random.default_rng(0)
    W = basis_rng.normal(scale=gamma, size=(NF, x.shape[1]))
    b = basis_rng.uniform(0, 2 * np.pi, NF)
    A = np.sqrt(2 / NF) * np.cos(x[tr] @ W.T + b)
    T = np.sqrt(2 / NF) * np.cos(x[te] @ W.T + b)
    print('Features computed; solving the full and adjusted ridge systems.', flush=True)
    (G, rhs) = (A.T @ A, A.T @ y[tr])
    w = cho_solve(cho_factor(G + len(tr) * LAM * np.eye(NF)), rhs)
    pred = A @ w
    test_mse = float(np.mean((T @ w - y[te]) ** 2))
    train_mse = float(np.mean((pred - y[tr]) ** 2))
    M = G + (len(tr) - 1) * LAM * np.eye(NF)
    factor = cho_factor(M)
    wt = cho_solve(factor, rhs)
    score_rng = np.random.default_rng(0)
    i1 = score_rng.integers(len(tr), size=20000)
    i2 = score_rng.integers(len(tr), size=20000)
    keep = i1 != i2
    (i1, i2) = (i1[keep], i2[keep])
    Lhat = float(np.percentile(np.abs(pred[i1] - pred[i2]) / np.maximum(np.linalg.norm(x[tr[i1]] - x[tr[i2]], axis=1), 1e-12), 99.5))
    epshat = float(np.quantile(np.abs(pred - y[tr]), 0.9))
    positions = {int(i): j for (j, i) in enumerate(tr)}
    summaries = {}
    all_validation = []
    for (request_seed, name) in [(0, 'diagnostic'), (1, 'operating_points')]:
        n_requests = config.diagnostic_requests if name == 'diagnostic' else 200
        requests = np.random.default_rng(request_seed).choice(tr, size=n_requests, replace=False)
        pos = np.array([positions[int(i)] for i in requests])
        targets = A[pos]
        solved = cho_solve(factor, targets.T)
        leverage = np.einsum('ij,ji->i', targets, solved)
        assert np.all((leverage >= 0) & (leverage < 1))
        residual = (targets @ wt - y[requests]) / (1 - leverage)
        floor = residual ** 2
        original = pred[pos] - y[requests]
        errors = []
        for k in [0, len(requests) // 2, len(requests) - 1]:
            v = targets[k]
            wd = np.linalg.solve(M - np.outer(v, v), rhs - v * y[requests[k]])
            errors.append(abs(float(v @ wd - y[requests[k]]) - float(residual[k])))
        assert max(errors) < 1e-08
        all_validation.extend(errors)
        sorted_tr = np.sort(tr)
        D = cdist(x[requests], x[sorted_tr])
        D[np.arange(len(requests)), np.searchsorted(sorted_tr, requests)] = np.inf
        nearest = sorted_tr[D.argmin(axis=1)]
        distance = D.min(axis=1)
        labelgap = np.abs(y[requests] - y[nearest])
        score = epshat + Lhat * distance + labelgap
        stratum = np.digitize(score, np.quantile(score, [1 / 3, 2 / 3]))
        rows = []
        for (k, i) in enumerate(requests):
            rows.append(dict(id=str(ids[i]), row_index=int(i), request_seed=request_seed, nearest_retained_id=str(ids[nearest[k]]), nn_distance=float(distance[k]), nn_label_gap=float(labelgap[k]), floor_score=float(score[k]), score_stratum=int(stratum[k]), adjusted_leverage=float(leverage[k]), original_residual=float(original[k]), retrained_residual=float(residual[k]), floor=float(floor[k]), floor_over_test_mse=float(floor[k] / test_mse), original_loss_over_floor=float(original[k] ** 2 / max(floor[k], 1e-12)), prediction_distance=float(abs(original[k] - residual[k])), test_mse=test_mse))
        write_csv(f'{name}_requests.csv', rows)
        bootstrap_rng = np.random.default_rng(20260906)
        strata = []
        for s in range(3):
            values = floor[stratum == s] / test_mse
            samples = bootstrap_rng.choice(values, size=(10000, len(values)))
            ci = np.quantile(np.median(samples, axis=1), [0.025, 0.975])
            strata.append(dict(stratum=s, n=len(values), median=float(np.median(values)), ci_low=float(ci[0]), ci_high=float(ci[1])))
        no_op = original ** 2 / np.maximum(floor, 1e-12)
        boot = bootstrap_rng.integers(len(floor), size=(10000, len(floor)))
        summaries[name] = dict(request_seed=request_seed, n_requests=len(requests), strata=strata, score_floor_spearman=float(spearmanr(score, floor).statistic), distance_floor_spearman=float(spearmanr(distance, floor).statistic), original_loss_over_floor_median=float(np.median(no_op)), original_loss_over_floor_ci95=np.quantile(np.median(no_op[boot], axis=1), [0.025, 0.975]).tolist(), prediction_distance_median=float(np.median(abs(original - residual))), clamped_floor_count=int(np.sum(floor < 1e-12)), maximum_direct_solve_residual_discrepancy=max(errors))
        print(f'{name}: computed {len(requests)} requests and checked three direct retained solves.', flush=True)
    out = dict(analysis='Algebraic analysis with fully recorded settings; conditional fixed-preprocessing reference.', corpus_size=len(y), training_size=len(tr), test_size=len(te), features=NF, lam=LAM, basis_seed=0, split_seed=0, gamma=float(gamma), train_mse=train_mse, test_mse=test_mse, subset_rule='first corpus_size rows in the existing cache; full cache when zero', empirical_lipschitz_quantile=0.995, empirical_lipschitz_pairs=20000, L_hat=Lhat, retained_fit_quantile=0.9, eps_hat=epshat, bootstrap_resamples=10000, bootstrap_seed=20260906, runs=summaries, maximum_validation_discrepancy=max(all_validation), dataset_sha256=hashlib.sha256(layout.descriptor_cache.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), elapsed_seconds=time.monotonic() - START)
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
