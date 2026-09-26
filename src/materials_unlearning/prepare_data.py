'Export materials structures or construct aligned graph caches.'
import argparse
import csv
import datetime
import json
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', required=True, choices=['mp', 'cifs', 'graphs'])
    p.add_argument('--out', required=True)
    p.add_argument('--corpus')
    p.add_argument('--cif-dir')
    p.add_argument('--targets')
    p.add_argument('--max-atoms', type=int, default=20)
    p.add_argument('--target', default='formation_energy_per_atom')
    p.add_argument('--mp20', action='store_true', help='apply the published MP-20 selection criteria (hull <= 0.08 eV/atom, formation energy <= 2 eV/atom) so the corpus is comparable to the MP-20 literature. Without this flag you get every entry under --max-atoms, a much larger and non-comparable set.')
    p.add_argument('--icsd-only', action='store_true', help='Restrict to experimentally reported structures; changes the default dataset selection.')
    a = p.parse_args()
    if a.source == 'graphs' and (not a.corpus):
        p.error('--source graphs requires --corpus')
    if a.source == 'cifs' and (not a.cif_dir or not a.targets):
        p.error('--source cifs requires --cif-dir and --targets')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    if a.source == 'mp':
        key = os.environ.get('MP_API_KEY')
        if not key:
            sys.exit('set MP_API_KEY (https://next-gen.materialsproject.org/api)')
        from mp_api.client import MPRester
        filters = dict(num_sites=(1, a.max_atoms))
        if a.mp20:
            filters.update(energy_above_hull=(None, 0.08), formation_energy=(None, 2.0))
        if a.icsd_only:
            filters.update(theoretical=False)
        with MPRester(key) as m:
            db_version = None
            try:
                db_version = m.get_database_version()
            except Exception as exc:
                print(f'WARNING: could not read the MP database version ({exc}). Dataset version metadata is unavailable.')
            docs = m.materials.summary.search(fields=['material_id', 'structure', 'formation_energy_per_atom', 'energy_above_hull', 'theoretical'], **filters)
        n = 0
        with open(a.out, 'w') as fh:
            for d in docs:
                if getattr(d, a.target, None) is None:
                    continue
                fh.write(json.dumps({'material_id': str(d.material_id), 'cif': d.structure.to(fmt='cif'), a.target: float(getattr(d, a.target)), 'energy_above_hull': None if getattr(d, 'energy_above_hull', None) is None else float(d.energy_above_hull), 'theoretical': bool(getattr(d, 'theoretical', True)), 'contributor': 'open' if not getattr(d, 'theoretical', True) else 'computed'}) + '\n')
                n += 1
        meta = {'n_records': n, 'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'mp_database_version': db_version, 'mp_api_version': getattr(__import__('mp_api'), '__version__', None), 'target': a.target, 'max_atoms': a.max_atoms, 'mp20_filter_applied': bool(a.mp20), 'icsd_only_filter_applied': bool(a.icsd_only), 'filters': {k: list(v) if isinstance(v, tuple) else v for (k, v) in filters.items()}, 'contributor_field': 'PROXY derived from the `theoretical` flag, not real licence or contributor metadata', 'published_mp20_count_for_reference': 45231}
        meta_path = os.path.splitext(a.out)[0] + '_snapshot.json'
        with open(meta_path, 'w') as fh:
            json.dump(meta, fh, indent=2)
        print(f'wrote {n} records to {a.out}')
        print(f'wrote snapshot metadata to {meta_path}')
        print(f'MP database version: {db_version}')
        if a.mp20 and n != 45231:
            print(f'NOTE: {n} records, not the published MP-20 count of 45231. The database has moved since MP-20 was defined. Use the recorded count and database version to identify this export.')
    elif a.source == 'cifs':
        tg = {}
        with open(a.targets) as fh:
            for row in csv.DictReader(fh):
                tg[row['id']] = float(row[a.target])
        n = 0
        with open(a.out, 'w') as fh:
            for f in sorted(os.listdir(a.cif_dir)):
                if not f.endswith('.cif'):
                    continue
                mid = os.path.splitext(f)[0]
                if mid not in tg:
                    continue
                fh.write(json.dumps({'material_id': mid, 'cif': open(os.path.join(a.cif_dir, f)).read(), a.target: tg[mid], 'contributor': 'open'}) + '\n')
                n += 1
        print(f'wrote {n} records to {a.out}')
    else:
        import numpy as np
        from pymatgen.core import Structure
        from .backends import CGCNNBackend
        recs = [json.loads(l) for l in open(a.corpus)]
        structs = [Structure.from_str(r['cif'], fmt='cif') for r in recs]
        g = CGCNNBackend.build_graphs(structs)
        ids = np.array([r.get('material_id', str(i)) for (i, r) in enumerate(recs)])
        np.savez_compressed(a.out, graphs=np.array(g, dtype=object), ids=ids)
        print(f'cached {len(g)} graphs to {a.out}')
        print(f"  mean atoms/structure {np.mean([len(x['z']) for x in g]):.1f} | mean edges/structure {np.mean([x['edge_index'].shape[1] for x in g]):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
