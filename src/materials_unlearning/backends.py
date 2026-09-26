"""Fixed-feature ridge and crystal graph neural network backends."""
from __future__ import annotations
import numpy as np

class _RidgeModel:

    def __init__(self, W, b, w, lam):
        (self.W, self.b, self.w, self.lam) = (W, b, w, lam)

    def _phi(self, X):
        X = np.atleast_2d(X)
        return np.sqrt(2.0 / self.W.shape[0]) * np.cos(X @ self.W.T + self.b)

    def predict(self, X):
        return self._phi(X) @ self.w

    def params(self):
        return self.w.copy()

    def set_params(self, v):
        self.w = np.asarray(v, dtype=float).copy()

    def grad(self, X, y):
        P = self._phi(X)
        r = P @ self.w - np.asarray(y, float)
        return 2.0 * P.T @ r / max(len(P), 1)

    def hessian(self, X):
        P = self._phi(X)
        return 2.0 * P.T @ P / max(len(P), 1) + 2.0 * self.lam * np.eye(P.shape[1])

class RidgeBackend:
    name = 'ridge'

    def __init__(self, n_features=512, gamma=1.0, lam=0.001, seed=0):
        (self.n_features, self.gamma, self.lam, self.seed) = (n_features, gamma, lam, seed)
        self._W = self._b = None

    def _basis(self, dim):
        if self._W is None or self._W.shape[1] != dim:
            r = np.random.default_rng(self.seed)
            self._W = r.normal(scale=self.gamma, size=(self.n_features, dim))
            self._b = r.uniform(0, 2 * np.pi, size=self.n_features)
        return (self._W, self._b)

    def fit(self, X, y, seed=0):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        (W, b) = self._basis(X.shape[1])
        m = _RidgeModel(W, b, np.zeros(self.n_features), self.lam)
        P = m._phi(X)
        A = P.T @ P + self.lam * len(P) * np.eye(P.shape[1])
        m.w = np.linalg.solve(A, P.T @ y)
        return m

class CGCNNBackend:
    name = 'cgcnn'

    def __init__(self, graphs=None, epochs=60, lr=0.01, hidden=64, n_conv=3, batch=128, device=None, weight_decay=0.0):
        import torch
        self.torch = torch
        self.graphs = graphs
        (self.epochs, self.lr, self.hidden, self.n_conv) = (epochs, lr, hidden, n_conv)
        (self.batch, self.weight_decay) = (batch, weight_decay)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    @staticmethod
    def build_graphs(structs, cutoff=8.0, max_nbr=12, n_gauss=41):
        import numpy as np
        centers = np.linspace(0, cutoff, n_gauss)
        var = centers[1] - centers[0]
        out = []
        for s in structs:
            nbrs = s.get_all_neighbors(cutoff)
            (idx_i, idx_j, dists) = ([], [], [])
            for (i, nl) in enumerate(nbrs):
                nl = sorted(nl, key=lambda x: x[1])[:max_nbr]
                for n in nl:
                    idx_i.append(i)
                    idx_j.append(n.index)
                    dists.append(n.nn_distance)
            if not idx_i:
                (idx_i, idx_j, dists) = ([0], [0], [cutoff])
            d = np.asarray(dists)
            ef = np.exp(-(d[:, None] - centers[None, :]) ** 2 / var ** 2)
            z = np.array([site.specie.Z if hasattr(site.specie, 'Z') else site.specie.element.Z for site in s])
            out.append(dict(z=z.astype(np.int64), edge_index=np.array([idx_i, idx_j], dtype=np.int64), edge_attr=ef.astype(np.float32)))
        return out

    def _net(self, seed):
        torch = self.torch
        import torch.nn as nn
        g = torch.Generator().manual_seed(int(seed))
        torch.manual_seed(int(seed))

        class ConvLayer(nn.Module):

            def __init__(self, h, ef):
                super().__init__()
                self.fc = nn.Linear(2 * h + ef, 2 * h)
                self.bn = nn.BatchNorm1d(2 * h)
                self.h = h

            def forward(self, x, ei, ea):
                (src, dst) = (ei[0], ei[1])
                z = torch.cat([x[src], x[dst], ea], dim=1)
                z = self.bn(self.fc(z))
                (f, c) = z.chunk(2, dim=1)
                msg = torch.sigmoid(f) * torch.nn.functional.softplus(c)
                agg = torch.zeros_like(x).index_add_(0, src, msg)
                return torch.nn.functional.softplus(x + agg)

        class Net(nn.Module):

            def __init__(self, h, n_conv, ef, n_z=100):
                super().__init__()
                self.emb = nn.Embedding(n_z, h)
                self.convs = nn.ModuleList([ConvLayer(h, ef) for _ in range(n_conv)])
                self.out = nn.Sequential(nn.Linear(h, h), nn.Softplus(), nn.Linear(h, 1))

            def forward(self, batch):
                (z, ei, ea, seg, nb) = batch
                x = self.emb(z)
                for c in self.convs:
                    x = c(x, ei, ea)
                pooled = torch.zeros(nb, x.shape[1], dtype=x.dtype, device=x.device)
                pooled = pooled.index_add(0, seg, x)
                cnt = torch.zeros(nb, 1, dtype=x.dtype, device=x.device)
                cnt = cnt.index_add(0, seg, torch.ones(len(seg), 1, dtype=x.dtype, device=x.device))
                return self.out(pooled / cnt).squeeze(-1)
        ef = self.graphs[0]['edge_attr'].shape[1]
        return Net(self.hidden, self.n_conv, ef).to(self.device)

    def _cache(self):
        if getattr(self, '_dev_cache', None) is None:
            torch = self.torch
            self._dev_cache = [(torch.as_tensor(g['z'], device=self.device), torch.as_tensor(g['edge_index'], device=self.device), torch.as_tensor(g['edge_attr'], device=self.device), len(g['z'])) for g in self.graphs]
        return self._dev_cache

    def _collate(self, idx):
        torch = self.torch
        cache = self._cache()
        (zs, eis, eas, segs, off) = ([], [], [], [], 0)
        for (k, i) in enumerate(idx):
            (z, ei, ea, na) = cache[int(i)]
            zs.append(z)
            eis.append(ei + off)
            eas.append(ea)
            segs.append(torch.full((na,), k, dtype=torch.long, device=self.device))
            off += na
        return (torch.cat(zs), torch.cat(eis, dim=1), torch.cat(eas), torch.cat(segs), len(idx))

    def _to_torch(self, idx):
        return self._collate(idx)

    def fit(self, X, y, seed=0):
        torch = self.torch
        idx = np.asarray(X[:, 0], dtype=int) if X.ndim > 1 else np.asarray(X, dtype=int)
        yt = torch.as_tensor(np.asarray(y, np.float32), device=self.device)
        net = self._net(seed)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        rng = np.random.default_rng(seed)
        net.train()
        for _ in range(self.epochs):
            perm = rng.permutation(len(idx))
            for s in range(0, len(perm), self.batch):
                sel = perm[s:s + self.batch]
                if len(sel) < 2:
                    continue
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(self._collate(idx[sel])), yt[sel])
                loss.backward()
                opt.step()
            sched.step()
        return _TorchModel(net, self, torch)

    def predict_batched(self, net, idx, chunk=512):
        torch = self.torch
        out = []
        net.eval()
        with torch.no_grad():
            for s in range(0, len(idx), chunk):
                out.append(net(self._collate(idx[s:s + chunk])).reshape(-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)

class _TorchModel:

    def __init__(self, net, backend, torch):
        (self.net, self.b, self.torch) = (net, backend, torch)

    def predict(self, X):
        idx = np.asarray(X[:, 0], dtype=int) if X.ndim > 1 else np.asarray(X, dtype=int)
        return self.b.predict_batched(self.net, idx)

    def params(self):
        return self.torch.nn.utils.parameters_to_vector(self.net.parameters()).detach().cpu().numpy()

    def set_params(self, v):
        self.torch.nn.utils.vector_to_parameters(self.torch.as_tensor(v, dtype=self.torch.float32, device=self.b.device), self.net.parameters())

    def grad(self, X, y):
        torch = self.torch
        idx = np.asarray(X[:, 0], dtype=int) if X.ndim > 1 else np.asarray(X, dtype=int)
        yt = torch.as_tensor(np.asarray(y, np.float32), device=self.b.device)
        self.net.train()
        self.net.zero_grad()
        torch.nn.functional.mse_loss(self.net(self.b._collate(idx)), yt).backward()
        return torch.cat([p.grad.reshape(-1) for p in self.net.parameters()]).detach().cpu().numpy()

def make_backend(name, **kw):
    if name == 'ridge':
        return RidgeBackend(**{k: v for (k, v) in kw.items() if k in ('n_features', 'gamma', 'lam', 'seed')})
    if name == 'cgcnn':
        return CGCNNBackend(**{k: v for (k, v) in kw.items() if k in ('graphs', 'epochs', 'lr', 'hidden', 'n_conv', 'batch', 'device', 'weight_decay')})
    raise ValueError(f'unknown backend {name}')
