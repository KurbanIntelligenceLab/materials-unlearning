'Cache structure descriptors and targets from a JSONL corpus.'
import argparse
import json
import os
import sys
import time
import numpy as np
from . import core


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--target', default='formation_energy_per_atom')
    p.add_argument('--limit', type=int, default=None, help='deterministic random subsample of this many structures')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    t0 = time.time()
    if a.limit:
        with open(a.corpus) as fh:
            lines = fh.readlines()
        rng = np.random.default_rng(a.seed)
        keep = rng.choice(len(lines), size=min(a.limit, len(lines)), replace=False)
        keep.sort()
        tmp = a.out + '.subsample.jsonl'
        with open(tmp, 'w') as fh:
            for i in keep:
                fh.write(lines[i])
        src = tmp
        print(f'subsampled {len(keep)} of {len(lines)} records (seed {a.seed})')
    else:
        src = a.corpus
        tmp = None
    c = core.load_corpus(src, target=a.target)
    core.cache_corpus(c, a.out)
    if tmp:
        os.remove(tmp)
    n = len(c['y'])
    X = c['X']
    print(f'cached {n} structures to {a.out} in {time.time() - t0:.0f}s')
    print(f'  descriptor dim   {X.shape[1]}')
    print(f"  distinct prototypes {len(set(c['proto'].tolist()))}")
    print(f"  target range     [{c['y'].min():.3f}, {c['y'].max():.3f}] eV/atom")
    print(f'  non-finite descriptor entries: {int((~np.isfinite(X)).sum())}')
    if not np.isfinite(X).all():
        bad = np.unique(np.argwhere(~np.isfinite(X))[:, 1])
        sys.exit(f'ERROR: non-finite values in descriptor columns {bad.tolist()[:20]}. Fix featurise() before measuring anything.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
