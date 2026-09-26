'Compare saved ridge predictions on the same deletion requests.'
from pathlib import Path
from .configuration import RepositoryPaths
import argparse
import csv
import hashlib
import json
import numpy as np


def main() -> int:
    layout = RepositoryPaths()
    REPO = layout.root
    RESULTS = layout.result()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('development/runs/paired'), help='Directory for newly computed results')
    args = parser.parse_args()
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open() as stream:
            return list(csv.DictReader(stream))
    def column(records: list[dict[str, str]], name: str) -> np.ndarray:
        return np.array([float(row[name]) for row in records])
    paths = [RESULTS / 'ridge_standard/operating_points_requests.csv', RESULTS / 'ridge_high_capacity/operating_points_requests.csv']
    records = [read_rows(path) for path in paths]
    ids = [[row['id'] for row in group] for group in records]
    assert ids[0] == ids[1] and len(set(ids[0])) == 200
    data = [{key: column(group, key) for key in ['floor', 'prediction_distance', 'original_residual', 'retrained_residual', 'adjusted_leverage']} for group in records]
    for group in data:
        assert np.allclose(group['floor'], group['retrained_residual'] ** 2, rtol=1e-13)
        assert np.allclose(group['prediction_distance'], np.abs(group['original_residual'] - group['retrained_residual']), rtol=1e-13)
        assert np.all((group['adjusted_leverage'] >= 0) & (group['adjusted_leverage'] < 1))
    metrics = []
    for group in data:
        metrics.append({'floor_median': float(np.median(group['floor'])), 'original_absolute_residual_median': float(np.median(np.abs(group['original_residual']))), 'adjusted_leverage_median': float(np.median(group['adjusted_leverage'])), 'prediction_distance_median': float(np.median(group['prediction_distance'])), 'prediction_distance_p90': float(np.quantile(group['prediction_distance'], 0.9))})
    summary = {'source_sha256': {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, 'requests': 200, 'identical_target_ids_in_order': True, 'lower_floor_at_8192': int(np.sum(data[1]['floor'] < data[0]['floor'])), 'larger_prediction_change_at_8192': int(np.sum(data[1]['prediction_distance'] > data[0]['prediction_distance'])), 'settings': {'2048': metrics[0], '8192': metrics[1]}}
    bootstrap = np.random.default_rng(20260906).integers(0, 200, size=(10000, 200))
    summary['paired_differences_8192_minus_2048'] = {key: {'median': float(np.median(data[1][key] - data[0][key])), 'ci95': np.quantile(np.median((data[1][key] - data[0][key])[bootstrap], axis=1), [0.025, 0.975]).tolist()} for key in ['floor', 'prediction_distance']}
    summary['lower_floor_and_larger_change_at_8192'] = int(np.sum((data[1]['floor'] < data[0]['floor']) & (data[1]['prediction_distance'] > data[0]['prediction_distance'])))
    assert summary['lower_floor_at_8192'] == 132
    assert summary['larger_prediction_change_at_8192'] == 172
    rng = np.random.default_rng(20260906)
    errors = []
    for (n, p) in [(2, 1), (2, 5), (7, 3), (7, 12), (30, 9)]:
        for lam in [0.0001, 0.01, 0.3, 2.0]:
            a = rng.normal(size=(n, p))
            y = rng.normal(size=n)
            m = a.T @ a + (n - 1) * lam * np.eye(p)
            full = np.linalg.solve(m + lam * np.eye(p), a.T @ y)
            adjusted = np.linalg.solve(m, a.T @ y)
            for i in [0, n - 1]:
                row = a[i]
                keep = np.arange(n) != i
                retained = np.linalg.solve(a[keep].T @ a[keep] + (n - 1) * lam * np.eye(p), a[keep].T @ y[keep])
                leverage = row @ np.linalg.solve(m, row)
                rhs = lam * row @ np.linalg.solve(m, full) + leverage * (row @ adjusted - y[i]) / (1 - leverage)
                errors.append(abs(float(row @ (retained - full) - rhs)))
    assert max(errors) < 1e-09
    summary['prediction_decomposition_checks'] = len(errors)
    summary['maximum_decomposition_error'] = max(errors)
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
