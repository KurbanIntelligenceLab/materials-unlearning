"""Retraining references and approximate parameter-update methods."""
from __future__ import annotations
import numpy as np

def _detached(original, backend, X, y, seed):
    if original is None:
        return backend.fit(X, y, seed=seed)
    try:
        import copy
        m = copy.deepcopy(original)
        assert m is not original
        return m
    except Exception:
        return backend.fit(X, y, seed=seed)

def _keep_mask(n, forget_idx):
    m = np.ones(n, bool)
    m[np.asarray(forget_idx)] = False
    return m

def retrain(backend, X, y, forget_idx, seed=0, **kw):
    k = _keep_mask(len(X), forget_idx)
    return backend.fit(X[k], y[k], seed=seed)

def do_nothing(backend, X, y, forget_idx, seed=0, original=None, **kw):
    if original is None:
        original = backend.fit(X, y, seed=seed)
    return original

def gradient_ascent(backend, X, y, forget_idx, seed=0, original=None, steps=30, lr=0.01, **kw):
    m = _detached(original, backend, X, y, seed)
    f = np.asarray(forget_idx)
    w = m.params().copy()
    for _ in range(steps):
        m.set_params(w)
        w = w + lr * m.grad(X[f], y[f])
    m.set_params(w)
    return m

def neggrad_plus(backend, X, y, forget_idx, seed=0, original=None, steps=30, lr=0.01, alpha=0.5, **kw):
    m = _detached(original, backend, X, y, seed)
    f = np.asarray(forget_idx)
    k = np.flatnonzero(_keep_mask(len(X), forget_idx))
    rng = np.random.default_rng(seed)
    w = m.params().copy()
    for _ in range(steps):
        m.set_params(w)
        sub = rng.choice(k, size=min(256, len(k)), replace=False)
        w = w + lr * (alpha * m.grad(X[f], y[f]) - (1 - alpha) * m.grad(X[sub], y[sub]))
    m.set_params(w)
    return m

def retain_finetune(backend, X, y, forget_idx, seed=0, original=None, steps=60, lr=0.01, **kw):
    m = _detached(original, backend, X, y, seed)
    k = np.flatnonzero(_keep_mask(len(X), forget_idx))
    rng = np.random.default_rng(seed)
    w = m.params().copy()
    for _ in range(steps):
        m.set_params(w)
        sub = rng.choice(k, size=min(256, len(k)), replace=False)
        w = w - lr * m.grad(X[sub], y[sub])
    m.set_params(w)
    return m

def scrub(backend, X, y, forget_idx, seed=0, original=None, steps=40, lr=0.01, alpha=0.3, **kw):
    teacher = original or backend.fit(X, y, seed=seed)
    t_pred_all = teacher.predict(X)
    m = backend.fit(X, y, seed=seed)
    f = np.asarray(forget_idx)
    k = np.flatnonzero(_keep_mask(len(X), forget_idx))
    rng = np.random.default_rng(seed)
    w = m.params().copy()
    for t in range(steps):
        m.set_params(w)
        if t % 2 == 0:
            w = w + lr * alpha * m.grad(X[f], y[f])
        else:
            sub = rng.choice(k, size=min(256, len(k)), replace=False)
            g_task = m.grad(X[sub], y[sub])
            g_dist = m.grad(X[sub], t_pred_all[sub])
            w = w - lr * (g_task + g_dist)
    m.set_params(w)
    return m

def influence_deletion(backend, X, y, forget_idx, seed=0, original=None, damping=0.01, **kw):
    m = _detached(original, backend, X, y, seed)
    f = np.asarray(forget_idx)
    g_f = m.grad(X[f], y[f])
    if hasattr(m, 'hessian'):
        k = _keep_mask(len(X), forget_idx)
        H = m.hessian(X[k])
        step = np.linalg.solve(H + damping * np.eye(len(H)), g_f)
    else:
        step = g_f / (damping + np.abs(g_f).mean() + 1e-12)
    m.set_params(m.params() + step * len(f) / len(X))
    return m

def deep_regression_unlearning(backend, X, y, forget_idx, seed=0, original=None, steps=40, lr=0.01, **kw):
    m = _detached(original, backend, X, y, seed)
    f = np.asarray(forget_idx)
    k = np.flatnonzero(_keep_mask(len(X), forget_idx))
    rng = np.random.default_rng(seed)
    y_fake = y.copy()
    y_fake[f] = rng.choice(y[k], size=len(f))
    w = m.params().copy()
    for _ in range(steps):
        m.set_params(w)
        sub = np.concatenate([rng.choice(k, size=min(256, len(k)), replace=False), f])
        w = w - lr * m.grad(X[sub], y_fake[sub])
    m.set_params(w)
    return m
METHODS = {'retrain (reference)': retrain, 'do nothing': do_nothing, 'gradient ascent': gradient_ascent, 'NegGrad+': neggrad_plus, 'retain fine-tuning': retain_finetune, 'SCRUB': scrub, 'influence / Newton': influence_deletion, 'deep regression unlearning': deep_regression_unlearning}



