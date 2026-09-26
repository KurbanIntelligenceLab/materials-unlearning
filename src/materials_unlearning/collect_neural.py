'Combine deletion-request shards and summarize their strata.'
import argparse
import csv
import glob
import json
import os
import numpy as np
from . import core


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dir', required=True)
    a = ap.parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.dir, 'neural_shard*.csv'))):
        with open(f) as fh:
            rows.extend(list(csv.DictReader(fh)))
    if not rows:
        raise SystemExit(f'no shard CSVs in {a.dir}')
    (seen, uniq) = ({}, [])
    for r in rows:
        if r['id'] in seen:
            previous = seen[r['id']]
            if any(previous.get(key) != value for key, value in r.items() if key != 'seconds'):
                raise SystemExit(f"conflicting duplicate request ID: {r['id']}")
            continue
        seen[r['id']] = r
        uniq.append(r)
    ratio = np.array([float(r['floor_over_testmse']) for r in uniq])
    strat = np.array([r['stratum'] for r in uniq])
    test_mse = float(uniq[0]['test_mse'])
    r2 = float(uniq[0]['r2'])
    loo = np.array([float(r['loo_floor']) for r in uniq if r.get('loo_floor')])
    print(f"{len(uniq)} requests from {len(glob.glob(os.path.join(a.dir, 'neural_shard*.csv')))} shard file(s) | test MSE {test_mse:.5f} | R2 {r2:.3f}")
    if r2 < 0.5:
        print('WARNING: R2 below 0.5 -- an undertrained model inflates the floor. Do not report these numbers.')
    out = []
    print(f"\n{'stratum':>12} {'n':>5} {'median floor/testMSE':>22} {'95% CI':>24}")
    for nm in ['redundant', 'sparse', 'isolated']:
        v = ratio[strat == nm]
        if not len(v):
            continue
        (lo, hi) = core.bootstrap_ci(v)
        out.append(dict(stratum=nm, n=int(len(v)), median=float(np.median(v)), ci_lo=float(lo), ci_hi=float(hi)))
        print(f"{nm:>12} {len(v):>5} {np.median(v):>22.4f} {f'[{lo:.4f}, {hi:.4f}]':>24}")
    if len(out) >= 2:
        r = out[-1]['median'] / max(out[0]['median'], 1e-12)
        sep = out[-1]['ci_lo'] > out[0]['ci_hi']
        print(f"\nisolated/redundant ratio: {r:.2f}x   CIs {('SEPARATE' if sep else 'OVERLAP')}")
        if not sep:
            print('The extreme-stratum confidence intervals overlap; this comparison does not establish separation.')
    if len(loo):
        print(f'\nleave-one-out control: median floor {np.median(loo) / test_mse:.4f} x test MSE over {len(loo)} requests')
        print('  The control deletes a DIFFERENT training point and reads the error at the request, so it measures how much the model moves for reasons unrelated to this deletion. A control comparable to the measured floors means the floor is not deletion-specific.')
    with open(os.path.join(a.dir, 'neural_summary.json'), 'w') as fh:
        json.dump(dict(n_requests=len(uniq), test_mse=test_mse, r2=r2, strata=out, loo_control_median_over_testmse=float(np.median(loo) / test_mse) if len(loo) else None), fh, indent=2)
    core.write_csv(os.path.join(a.dir, 'neural_strata.csv'), ['stratum', 'n', 'median_floor_over_testmse', 'ci_lo', 'ci_hi'], [(d['stratum'], d['n'], d['median'], d['ci_lo'], d['ci_hi']) for d in out])
    print(f'\nwrote {a.dir}/neural_strata.csv and neural_summary.json')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
