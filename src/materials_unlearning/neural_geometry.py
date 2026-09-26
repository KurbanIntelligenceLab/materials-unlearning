'Analyze nearest-neighbor descriptors against saved neural deletion losses.'
from pathlib import Path
from .configuration import RepositoryPaths
import argparse
import csv
import json
import sys
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr


def main() -> int:
    layout = RepositoryPaths()
    REPO = layout.root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('development/runs/neural'), help='Directory for newly computed results')
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    with (REPO / 'results/neural/deletion_losses.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with np.load(layout.descriptor_cache, allow_pickle=False) as z:
        x = z['X'].astype(float)
        y = z['y'].astype(float)
        ids = z['ids'].astype(str)
    perm = np.random.default_rng(0).permutation(len(y))
    tr = perm[int(0.2 * len(y)):]
    tr = tr[np.argsort(ids[tr])]
    x = (x - x[tr].mean(axis=0)) / (x[tr].std(axis=0) + 1e-09)
    request = np.array([int(r['idx']) for r in rows])
    assert all((ids[i] == r['id'] for (i, r) in zip(request, rows)))
    assert set(request).issubset(set(tr))
    assert len(set(ids)) == len(ids)
    d = cdist(x[request], x[tr])
    positions = {int(i): j for (j, i) in enumerate(tr)}
    for (k, i) in enumerate(request):
        d[k, positions[int(i)]] = np.inf
    nnpos = d.argmin(axis=1)
    nn = tr[nnpos]
    distance = d[np.arange(len(rows)), nnpos]
    gap = np.abs(y[request] - y[nn])
    floor = np.array([float(r['floor']) for r in rows])
    control = np.array([float(r['loo_floor']) for r in rows])
    test = np.array([float(r['test_mse']) for r in rows])
    rng = np.random.default_rng(20260905)
    rp = rng.permutation(len(y))
    inverse = np.argsort(rp)
    xp = x[rp]
    errors = []
    for k in [0, 49, 99, 149, 199]:
        direct = np.linalg.norm(xp[inverse[tr]] - xp[inverse[request[k]]], axis=1)
        direct[positions[int(request[k])]] = np.inf
        assert tr[direct.argmin()] == nn[k]
        errors.append(abs(float(direct.min() - distance[k])))
    assert max(errors) < 1e-10
    summary = {'analysis_status': 'exploratory descriptor analysis of single-seed losses', 'descriptor_columns': x.shape[1], 'requests': len(rows), 'max_independent_distance_error': max(errors), 'joint_row_permutation_checks': len(errors), 'full_FloorScore_recomputed': False}
    boot = rng.integers(len(rows), size=(2000, len(rows)))
    for (name, predictor) in [('nn_distance', distance), ('nn_label_gap', gap)]:
        rho = float(spearmanr(predictor, floor).statistic)
        bs = np.array([spearmanr(predictor[b], floor[b]).statistic for b in boot])
        summary[name] = {'spearman_vs_floor': rho, 'request_bootstrap_ci95': np.quantile(bs, [0.025, 0.975]).tolist(), 'spearman_vs_paired_loss_difference': float(spearmanr(predictor, floor - control).statistic)}
    cut = np.quantile(distance, [1 / 3, 2 / 3])
    strata = np.digitize(distance, cut)
    table = []
    for (k, name) in enumerate(['near', 'middle', 'far']):
        v = (floor / test)[strata == k]
        b = rng.integers(len(v), size=(5000, len(v)))
        ci = np.quantile(np.median(v[b], axis=1), [0.025, 0.975])
        table.append({'stratum': name, 'n': len(v), 'median_floor_over_test_mse': float(np.median(v)), 'ci95': ci.tolist()})
    summary['distance_strata'] = table
    summary['distance_stratum_cutpoints'] = cut.tolist()
    summary['far_over_near_median_ratio'] = table[2]['median_floor_over_test_mse'] / table[0]['median_floor_over_test_mse']
    summary['cautions'] = ['Existing floors condition on one training seed; bootstrap covers requests.', 'Original training-split normalization is frozen; this is descriptive geometry.', 'Distance and label gap are separate diagnostics, not a recalibrated FloorScore.', 'The analysis is exploratory and must not be presented as prespecified confirmation.']
    with (OUT / 'descriptor_requests.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'row_index', 'nearest_retained_id', 'nn_distance', 'nn_label_gap', 'floor', 'control_loss', 'test_mse', 'distance_stratum'])
        for (k, r) in enumerate(rows):
            writer.writerow([r['id'], request[k], ids[nn[k]], distance[k], gap[k], floor[k], control[k], test[k], table[strata[k]]['stratum']])
    (OUT / 'descriptor_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
