"""Build distribution files with neutral archive ownership and timestamps."""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sanitize_sdist(source: Path, destination: Path) -> None:
    with tarfile.open(source, 'r:gz') as original:
        with destination.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w', format=tarfile.PAX_FORMAT) as clean:
                for member in original:
                    if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                        raise ValueError(f'unsupported archive member: {member.name}')
                    if Path(member.name).is_absolute() or '..' in Path(member.name).parts:
                        raise ValueError('unsafe archive path')
                    contents = original.extractfile(member) if member.isfile() else None
                    member.uid = member.gid = member.mtime = 0
                    member.uname = member.gname = ''
                    member.pax_headers = {}
                    member.mode = 0o755 if member.isdir() else 0o644
                    clean.addfile(member, contents)
                    if contents is not None:
                        contents.close()


def sanitize_wheel(source: Path, destination: Path) -> None:
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as clean:
        for name in original.namelist():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('unsafe archive path')
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            entry.compress_type = zipfile.ZIP_DEFLATED
            clean.writestr(entry, original.read(name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'dist')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        subprocess.run([sys.executable, '-m', 'build', '--outdir', str(output)], cwd=ROOT, check=True)
        for archive in sorted(output.iterdir()):
            destination = args.out / archive.name
            if archive.name.endswith('.tar.gz'):
                sanitize_sdist(archive, destination)
            elif archive.name.endswith('.whl'):
                sanitize_wheel(archive, destination)
            else:
                raise ValueError(f'unexpected build artifact: {archive.name}')
            print(f'Built {destination.name} with neutral archive metadata.')


if __name__ == '__main__':
    main()
