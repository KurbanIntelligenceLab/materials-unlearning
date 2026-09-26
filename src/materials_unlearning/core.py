"""Numerical utilities and materials descriptors for deletion analysis."""
from __future__ import annotations
import json
import os
import hashlib
import warnings
import numpy as np

def load_corpus(path, target='formation_energy_per_atom', limit=None):
    path = os.fspath(path)
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    if path.endswith('.npz'):
        with np.load(path, allow_pickle=False) as z:
            corpus = {key: z[key] for key in ('ids', 'X', 'y', 'proto', 'spg', 'contributor')}
        n = len(corpus['y'])
        if any(len(value) != n for value in corpus.values()):
            raise ValueError('cached corpus fields must have matching row counts')
        return {key: value[:limit] for key, value in corpus.items()}
    recs = []
    with open(path) as fh:
        for (i, line) in enumerate(fh):
            if limit and i >= limit:
                break
            recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f'no records in {path}')
    from pymatgen.core import Structure
    structs = [Structure.from_str(r['cif'], fmt='cif') for r in recs]
    y = np.array([float(r[target]) for r in recs])
    ids = np.array([r.get('material_id', str(i)) for (i, r) in enumerate(recs)])
    contrib = np.array([r.get('contributor', 'open') for r in recs])
    X = np.array([featurise(s) for s in structs])
    (proto, spg) = prototype_labels(structs)
    return dict(ids=ids, X=X, y=y, proto=proto, spg=spg, contributor=contrib)

def cache_corpus(corpus, path):
    np.savez_compressed(path, **corpus)
ELEMENTS = None

def featurise(struct, n_rdf=32, rmax=12.0):
    global ELEMENTS
    from pymatgen.core.periodic_table import Element
    if ELEMENTS is None:
        ELEMENTS = [Element.from_Z(z).symbol for z in range(1, 95)]
    comp = np.zeros(len(ELEMENTS))
    for site in struct:
        sym = site.specie.element.symbol if hasattr(site.specie, 'element') else str(site.specie)
        if sym in ELEMENTS:
            comp[ELEMENTS.index(sym)] += 1
    comp /= max(comp.sum(), 1)
    d = struct.distance_matrix[np.triu_indices(len(struct), 1)]
    if d.size == 0:
        d = np.array([1.0])
    (hist, _) = np.histogram(d, bins=n_rdf, range=(0, rmax), density=True)
    geo = np.array([struct.volume ** (1 / 3) / max(len(struct), 1) ** (1 / 3), len(struct) / struct.volume, float(d.min()), float(d.mean()), float(d.std()), len(struct), struct.lattice.a, struct.lattice.c / max(struct.lattice.a, 1e-09)])
    return np.concatenate([comp, geo, hist])

def standardise(X):
    (mu, sd) = (X.mean(0), X.std(0) + 1e-09)
    return ((X - mu) / sd, mu, sd)

def prototype_labels(structs, symprec=0.1):
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    (proto, spg) = ([], [])
    for s in structs:
        try:
            sga = SpacegroupAnalyzer(s, symprec=symprec)
            n = sga.get_space_group_number()
        except Exception:
            n = 0
        try:
            anon = s.composition.anonymized_formula
        except Exception:
            anon = '?'
        spg.append(n)
        proto.append(f'{n}|{anon}')
    return (np.array(proto), np.array(spg))

def empirical_lipschitz(predict, X, n_pairs=20000, rng=None):
    rng = rng or np.random.default_rng(0)
    i = rng.integers(len(X), size=n_pairs)
    j = rng.integers(len(X), size=n_pairs)
    keep = i != j
    (i, j) = (i[keep], j[keep])
    num = np.abs(predict(X[i]) - predict(X[j]))
    den = np.linalg.norm(X[i] - X[j], axis=1)
    return float(np.percentile(num / np.maximum(den, 1e-12), 99.5))

def training_error_level(predict, X, y, q=0.9):
    return float(np.quantile(np.abs(predict(X) - y), q))



def floorprobe(x_f, y_f, X_retain, y_retain, L_hat, eps_bar):
    d = np.linalg.norm(X_retain - x_f, axis=1)
    j = int(np.argmin(d))
    return (eps_bar + L_hat * float(d[j]) + abs(float(y_f) - float(y_retain[j])), j, float(d[j]))

def stratify(probe_values, n_bins=3):
    qs = np.quantile(probe_values, np.linspace(0, 1, n_bins + 1)[1:-1])
    lab = np.digitize(probe_values, qs)
    names = np.array(['redundant', 'sparse', 'isolated'])[:n_bins]
    return (names[lab], names)



def measure_floor(backend, X, y, drop_idx, target_idx, seed=0, cap=None, train_idx=None):
    drop = np.asarray(drop_idx)
    assert target_idx in set(drop.tolist()), 'target must be among the deleted'
    if train_idx is None:
        warnings.warn('measure_floor called without train_idx: retraining on every row except the deleted ones, INCLUDING the held-out test split. The resulting number is not the deletion floor. Pass the training split.', RuntimeWarning, stacklevel=2)
        keep = np.ones(len(X), bool)
        keep[drop] = False
        keep_idx = np.flatnonzero(keep)
    else:
        keep_idx = np.setdiff1d(np.asarray(train_idx), drop, assume_unique=False)
    m = backend.fit(X[keep_idx], y[keep_idx], seed=seed)
    pred = float(m.predict(X[target_idx][None, :])[0])
    err = (pred - float(y[target_idx])) ** 2
    return min(err, cap) if cap is not None else err

def measure_displacement_control(backend, X, y, drop_idx, target_idx, seed=0, train_idx=None):
    drop = np.asarray(drop_idx)
    assert target_idx not in set(drop.tolist()), 'the control requires the target NOT be deleted; use measure_floor otherwise'
    if train_idx is None:
        keep = np.ones(len(X), bool)
        keep[drop] = False
        keep_idx = np.flatnonzero(keep)
    else:
        keep_idx = np.setdiff1d(np.asarray(train_idx), drop, assume_unique=False)
    m = backend.fit(X[keep_idx], y[keep_idx], seed=seed)
    pred = float(m.predict(X[target_idx][None, :])[0])
    return (pred - float(y[target_idx])) ** 2

def bootstrap_ci(vals, n=2000, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    v = np.asarray(vals)
    bs = np.array([np.median(rng.choice(v, v.size, replace=True)) for _ in range(n)])
    return (float(np.percentile(bs, 100 * alpha / 2)), float(np.percentile(bs, 100 * (1 - alpha / 2))))

def write_csv(path, header, rows):
    with open(path, 'w') as fh:
        fh.write(','.join(header) + '\n')
        for r in rows:
            fh.write(','.join((f'{v:.6g}' if isinstance(v, float) else str(v) for v in r)) + '\n')
    print(f'wrote {path}')

