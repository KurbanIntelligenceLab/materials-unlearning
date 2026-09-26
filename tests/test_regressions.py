"""Regression checks for corpus alignment, numerical references, and releases."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from materials_unlearning import core, unlearning, collect_neural
from materials_unlearning.backends import RidgeBackend
from materials_unlearning.cli.shared import load, parser, split

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_release', ROOT / 'tools/build_release.py')
build_release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_release)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / 'corpus.npz'
        self.corpus = dict(ids=np.array(['a', 'b', 'c']), X=np.arange(6).reshape(3, 2),
                           y=np.arange(3.), proto=np.array(['p'] * 3), spg=np.zeros(3),
                           contributor=np.array(['open'] * 3))
        np.savez(self.path, **self.corpus)

    def test_limit_applies_to_every_cached_field_and_accepts_path(self):
        for limit in [None, 1, 2, 8]:
            for path in [self.path, str(self.path)]:
                actual = core.load_corpus(path, limit=limit)
                for key, value in self.corpus.items():
                    np.testing.assert_array_equal(actual[key], value[:limit])

    def test_nonpositive_limit_rejected(self):
        for limit in [0, -1]:
            with self.assertRaises(ValueError):
                core.load_corpus(self.path, limit=limit)

    def test_cache_alignment_is_checked(self):
        self.corpus['ids'] = self.corpus['ids'][:1]
        np.savez(self.path, **self.corpus)
        with self.assertRaisesRegex(ValueError, 'row counts'):
            core.load_corpus(self.path)

    def test_descriptor_cache_does_not_unpickle_objects(self):
        self.corpus['ids'] = self.corpus['ids'].astype(object)
        np.savez(self.path, **self.corpus)
        with self.assertRaises(ValueError):
            core.load_corpus(self.path)

    def test_graph_prefix_remains_aligned(self):
        graphs = self.root / 'graphs.npz'
        np.savez(graphs, graphs=np.array([{'index': i} for i in range(3)], dtype=object), ids=self.corpus['ids'])
        args = parser('test').parse_args(['--corpus', str(self.path), '--graphs', str(graphs), '--limit', '2'])
        actual = load(args)
        self.assertEqual([g['index'] for g in actual['graphs']], [0, 1])
        np.testing.assert_array_equal(actual['ids'], ['a', 'b'])
        np.savez(graphs, graphs=np.array([{}] * 3, dtype=object), ids=self.corpus['ids'][::-1])
        with self.assertRaisesRegex(SystemExit, 'ids do not match'):
            load(args)

    def test_positive_cli_controls(self):
        for flag in ['--limit', '--seeds', '--n-requests', '--n-features', '--epochs']:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser('test').parse_args([flag, '0'])


class NumericalTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(13)
        self.X, self.y = rng.normal(size=(18, 3)), rng.normal(size=18)
        self.backend = RidgeBackend(n_features=9, lam=0.03)

    def test_ridge_matches_augmented_least_squares(self):
        model = self.backend.fit(self.X, self.y)
        phi = model._phi(self.X)
        augmented = np.vstack([phi, np.sqrt(len(self.y) * model.lam) * np.eye(phi.shape[1])])
        targets = np.r_[self.y, np.zeros(phi.shape[1])]
        expected = np.linalg.lstsq(augmented, targets, rcond=None)[0]
        np.testing.assert_allclose(model.params(), expected, rtol=1e-10, atol=1e-12)

    def test_deletion_reference_uses_only_retained_training_rows(self):
        train, test = split(len(self.y), 0)
        target = int(train[0])
        expected = self.backend.fit(self.X[train[1:]], self.y[train[1:]])
        error = float((expected.predict(self.X[[target]])[0] - self.y[target]) ** 2)
        actual = core.measure_floor(self.backend, self.X, self.y, [target], target, train_idx=train)
        self.assertAlmostEqual(actual, error, places=11)
        changed = self.y.copy()
        changed[test] += 1e6
        isolated = core.measure_floor(self.backend, self.X, changed, [target], target, train_idx=train)
        self.assertAlmostEqual(actual, isolated, places=11)

    def test_updates_leave_original_model_unchanged(self):
        original = self.backend.fit(self.X, self.y)
        before = original.params().copy()
        for method in unlearning.METHODS.values():
            model = method(self.backend, self.X, self.y, [0], original=original, steps=2)
            self.assertTrue(np.isfinite(model.predict(self.X)).all())
            np.testing.assert_array_equal(original.params(), before)

    def test_data_gradient_matches_finite_difference(self):
        model = self.backend.fit(self.X, self.y)
        initial = model.params()
        gradient = model.grad(self.X[:3], self.y[:3])
        numerical = np.zeros_like(initial)
        for i in range(len(initial)):
            delta = np.zeros_like(initial)
            delta[i] = 1e-6
            model.set_params(initial + delta)
            plus = np.mean((model.predict(self.X[:3]) - self.y[:3]) ** 2)
            model.set_params(initial - delta)
            minus = np.mean((model.predict(self.X[:3]) - self.y[:3]) ** 2)
            numerical[i] = (plus - minus) / 2e-6
        np.testing.assert_allclose(gradient, numerical, atol=1e-9)


class CollectionTests(unittest.TestCase):
    def test_conflicting_duplicate_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i, floor in enumerate(['1', '2']):
                (root / f'neural_shard{i}.csv').write_text('id,floor\na,' + floor + '\n')
            with patch.object(sys, 'argv', ['collect-neural', '--dir', str(root)]):
                with self.assertRaisesRegex(SystemExit, 'conflicting duplicate'):
                    collect_neural.main()


class ArchiveTests(unittest.TestCase):
    def test_source_archive_strips_identity_metadata_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, target = root / 'source.tar.gz', root / 'clean.tar.gz'
            with tarfile.open(source, 'w:gz') as archive:
                info = tarfile.TarInfo('project/README.md')
                info.size, info.uid, info.gid = 5, 501, 20
                info.uname, info.gname, info.mtime = 'local-user', 'local-group', 123456789
                info.pax_headers = {'mtime': '123456789.123', 'comment': 'local-user'}
                archive.addfile(info, io.BytesIO(b'hello'))
            build_release.sanitize_sdist(source, target)
            with tarfile.open(target) as archive:
                info = archive.getmembers()[0]
                self.assertEqual((info.uid, info.gid, info.uname, info.gname, info.mtime), (0, 0, '', '', 0))
                self.assertEqual(info.pax_headers, {})
                self.assertEqual(archive.extractfile(info).read(), b'hello')
            self.assertEqual(target.read_bytes()[4:8], b'\0' * 4)

    def test_wheel_metadata_is_neutral_and_contents_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with zipfile.ZipFile(root / 'source.whl', 'w') as archive:
                info = zipfile.ZipInfo('package/__init__.py', (2026, 9, 26, 12, 0, 0))
                info.comment = b'local-user'
                archive.writestr(info, b'example')
            build_release.sanitize_wheel(root / 'source.whl', root / 'clean.whl')
            with zipfile.ZipFile(root / 'clean.whl') as archive:
                info = archive.infolist()[0]
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual((info.extra, info.comment, archive.comment), (b'', b'', b''))
                self.assertEqual(archive.read(info), b'example')


if __name__ == '__main__':
    unittest.main()
