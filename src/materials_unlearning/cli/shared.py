"""Shared CLI utilities for script modules."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import core
from .. import backends


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--corpus", default="data/mp20_full.npz", help="jsonl or cached .npz; see README.md")
    p.add_argument(
        "--graphs",
        default=None,
        help=(
            "cached CGCNN graphs .npz (from prepare_data.py --source graphs). REQUIRED by --backend cgcnn when "
            "--corpus is a descriptor .npz, since the descriptor cache does not carry graphs. Must be built from the same "
            "jsonl, in the same row order, or the graphs and targets will be misaligned."
        ),
    )
    p.add_argument("--backend", default="ridge", choices=["ridge", "cgcnn"])
    p.add_argument("--target", default="formation_energy_per_atom")
    p.add_argument("--limit", type=positive_int, default=None)
    p.add_argument("--n-requests", type=positive_int, default=200)
    p.add_argument("--seeds", type=positive_int, default=5)
    p.add_argument("--out", default="development/runs")
    p.add_argument("--epochs", type=positive_int, default=60)
    p.add_argument("--n-features", type=positive_int, default=2048, help="Number of random Fourier features for the ridge backend (default: 2048).")
    p.add_argument("--lam", type=float, default=1e-06, help="ridge backend: ridge penalty")
    p.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="ridge backend: RBF bandwidth. Default is the median pairwise-distance heuristic on the standardized training split.",
    )
    p.add_argument("--demo", action="store_true", help="synthetic stand-in corpus so the pipeline can be exercised without MP-20; NEVER report these numbers")
    return p


def load(args: argparse.Namespace):
    if args.demo:
        rng = np.random.default_rng(0)
        (n_fam, per_fam, d, nf) = (150, 6, 24, 256)
        n = n_fam * per_fam
        C = rng.normal(size=(n_fam, d)) * 1.5
        X = np.repeat(C, per_fam, axis=0) + rng.normal(size=(n, d)) * 0.05
        W = rng.normal(size=(nf, d))
        b = rng.uniform(0, 2 * np.pi, nf)
        y = np.sqrt(2 / nf) * np.cos(X @ W.T + b) @ (rng.normal(size=nf) * 0.3) + rng.normal(size=n) * 0.01
        proto = np.array([f"P{i // per_fam}" for i in range(n)])
        return dict(ids=np.arange(n).astype(str), X=X, y=y, proto=proto, spg=np.zeros(n, int), contributor=rng.choice(["open", "licensed", "partner"], n))

    if not os.path.exists(args.corpus):
        raise SystemExit(f"corpus not found: {args.corpus}\nSee README.md, or pass --demo to smoke-test the pipeline on a synthetic stand-in.")

    corpus = core.load_corpus(args.corpus, target=args.target, limit=args.limit)
    if getattr(args, "graphs", None):
        if not os.path.exists(args.graphs):
            raise SystemExit(f"graphs cache not found: {args.graphs}")

        # Graph caches contain dictionaries; only load locally generated, trusted files.
        with np.load(args.graphs, allow_pickle=True) as graph_data:
            g = list(graph_data["graphs"][:args.limit])
            graph_ids = graph_data["ids"][:args.limit] if "ids" in graph_data.files else None

        if len(g) != len(corpus["y"]):
            raise SystemExit(f"graph/corpus length mismatch: {len(g)} graphs vs {len(corpus['y'])} corpus rows. Both must be built from the same jsonl in the same order; rebuild the graphs from the corpus you are using.")

        if graph_ids is not None:
            if not np.array_equal(np.asarray(graph_ids), np.asarray(corpus["ids"])):
                raise SystemExit("graph ids do not match corpus ids in order. Rebuild the graph cache from this corpus.")

        corpus["graphs"] = g

    return corpus


def make_backend(args: argparse.Namespace, corpus: dict[str, Any]):
    if args.backend == "ridge":
        X = np.asarray(corpus["X"], float)
        (tr, _) = split(len(corpus["y"]), 0)
        mu = X[tr].mean(0)
        sd = X[tr].std(0) + 1e-09
        Xs = (X - mu) / sd
        gamma = getattr(args, "gamma", None) or 1.0 / _median_distance(Xs[tr])
        n_feat = getattr(args, "n_features", None) or 512
        lam = getattr(args, "lam", None)
        lam = 1e-06 if lam is None else lam
        backend = backends.make_backend("ridge", n_features=n_feat, lam=lam, gamma=gamma)
        return (backend, Xs)

    graphs = corpus.get("graphs")
    if graphs is None:
        raise SystemExit("cgcnn backend needs cached graphs; run prepare_data.py first")
    return (backends.make_backend("cgcnn", graphs=graphs, epochs=args.epochs), np.arange(len(corpus["y"]))[:, None].astype(float))


def _median_distance(X: np.ndarray, n_pairs: int = 4000, seed: int = 0) -> float:
    r = np.random.default_rng(seed)
    i = r.integers(len(X), size=n_pairs)
    j = r.integers(len(X), size=n_pairs)
    d = np.linalg.norm(X[i] - X[j], axis=1)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 1.0


def split(n: int, seed: int = 0, frac: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return (idx[k:], idx[:k])
