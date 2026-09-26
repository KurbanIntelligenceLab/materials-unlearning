#!/usr/bin/env python3
"""Check and clean the public release surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOTS = (ROOT / "src", ROOT / "results", ROOT / "tools", ROOT / "proofs", ROOT / "tests")
PUBLIC_FILES = (
    ROOT / ".gitignore",
    ROOT / ".gitattributes",
    ROOT / "LICENSE",
    ROOT / "Makefile",
    ROOT / "MANIFEST.in",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "RELEASE_PLAN.md",
    ROOT / "pyproject.toml",
)
TEXT_SUFFIXES = {
    ".bib",
    ".cfg",
    ".csv",
    ".json",
    ".lean",
    ".md",
    ".py",
    ".sty",
    ".tex",
    ".toml",
    ".txt",
}
PRIVATE_PATHS = re.compile(r"(?:/Users|/home)/[A-Za-z0-9_.-]+|[A-Z]:\\Users\\[A-Za-z0-9_.-]+")
SECRET_PATTERN = re.compile(
    r"(?:AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|(?:api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][A-Za-z0-9_/-]{16,})",
    re.IGNORECASE,
)
PRIVATE_COMPONENTS = {"data", "development", ".lake", ".venv", "__pycache__", ".DS_Store"}

INTERNAL_NOTE = re.compile(
    r"\b(?:co-?author|development note|internal note|review comment|reviewer)\b"
    r"|^[^\n]*#.*\b(?:TODO|FIXME|XXX)\b",
    re.IGNORECASE | re.MULTILINE,
)
RETIRED_DIRECTORIES = (
    "dist",
    "figure1_blue_yellow_red_20260909",
    "figure1_energy_example_20260908",
    "figure1_grayscale_20260909",
    "figure1_soft_accents_20260909",
    "figure1_soft_gray_20260909",
    "final_package_sweep",
    "package_polish_backup",
    "structure_refactor_backup",
    "structure_sweep",
    "submission_before_package_refactor",
)


def public_text_files() -> Iterable[Path]:
    yield from (path for path in PUBLIC_FILES if path.is_file())
    for root in PUBLIC_ROOTS:
        for directory, children, names in os.walk(root):
            children[:] = [name for name in children if name not in {".lake", "__pycache__"} and not name.endswith(".egg-info")]
            for name in names:
                path = Path(directory) / name
                if path.suffix.lower() in TEXT_SUFFIXES or name in {"lean-toolchain", ".gitignore"}:
                    yield path


def release_errors(anonymous: bool = False) -> list[str]:
    errors: list[str] = []
    for required_file in PUBLIC_FILES:
        if not required_file.is_file():
            errors.append(f"Missing release file: {required_file.name}")
    if errors:
        return errors
    for path in public_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        relative = path.relative_to(ROOT)
        if SECRET_PATTERN.search(text):
            errors.append(f"{relative}: contains a possible embedded credential")
        if PRIVATE_PATHS.search(text):
            errors.append(f"{relative}: contains a private absolute path")
        if relative.parts[0] == "src" and INTERNAL_NOTE.search(text):
            errors.append(f"{relative}: contains an internal development marker")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for required in ("/data/", "/development/", "/final_manuscript/"):
        if required not in gitignore:
            errors.append(f".gitignore: missing {required}")

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    if anonymous and not re.search(r"^Copyright \(c\) \d{4} ANON$", license_text, re.MULTILINE):
        errors.append("LICENSE: anonymous release requires ANON as copyright holder")
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    if anonymous and re.search(r"(?m)^\s*(?:authors|maintainers)\s*=", metadata):
        errors.append("pyproject.toml: anonymous release contains author or maintainer metadata")
    for entry in json.loads((ROOT / "results" / "files.json").read_text()):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] not in {"results", "data"}:
            errors.append(f"Invalid evidence path: {relative}")
            continue
        path = ROOT / relative
        if not path.is_file():
            if relative.parts[0] != "data":
                errors.append(f"Missing evidence: {relative}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            errors.append(f"Evidence checksum mismatch: {relative}")
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True)
        if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve():
            tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
            for name in filter(None, tracked):
                path = ROOT / name
                if PRIVATE_COMPONENTS.intersection(Path(name).parts) or Path(name).name.startswith(".env"):
                    errors.append(f"{name}: private or generated file is tracked")
                elif path.is_symlink():
                    errors.append(f"{name}: release must not contain symlinks")
                elif path.is_file():
                    contents = path.read_text(encoding="utf-8", errors="replace")
                    if PRIVATE_PATHS.search(contents) or SECRET_PATTERN.search(contents):
                        errors.append(f"{name}: tracked file contains a private path or possible credential")
    except FileNotFoundError:
        pass
    return errors


def anonymous_history_errors() -> list[str]:
    try:
        history = subprocess.run(
            ["git", "log", "HEAD", "--format=%H%x00%an%x00%ae%x00%cn%x00%ce"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["Anonymous history audit requires a Git checkout with commits"]
    errors = []
    for line in history.stdout.splitlines():
        commit, author, email, committer, committer_email = line.split("\0")
        if (author, email, committer, committer_email) != ("ANON", "anon@example.invalid", "ANON", "anon@example.invalid"):
            errors.append(f"{commit[:12]}: identifying Git author or committer metadata")
        license_text = subprocess.run(["git", "show", f"{commit}:LICENSE"], cwd=ROOT, capture_output=True, text=True)
        if license_text.returncode or not re.search(r"^Copyright \(c\) \d{4} ANON$", license_text.stdout, re.MULTILINE):
            errors.append(f"{commit[:12]}: historical license is not anonymous")
    return errors


def check_release(anonymous: bool = False, history: bool = False) -> int:
    errors = release_errors(anonymous)
    if history:
        errors.extend(anonymous_history_errors())
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("PASS: release files, path hygiene, and available evidence checksums.")
    if anonymous:
        print("PASS: anonymous license and package metadata; audit Git identities separately.")
    return 0


def archive_retired() -> int:
    development = ROOT / "development" / "development_core"
    destination = development / "retired_results"
    destination.mkdir(parents=True, exist_ok=True)
    moved = 0
    for name in RETIRED_DIRECTORIES:
        source = development / name
        target = destination / name
        if not source.exists():
            continue
        if target.exists():
            print(f"SKIP: {target.relative_to(ROOT)} already exists")
            continue
        shutil.move(str(source), target)
        print(f"MOVED: {source.relative_to(ROOT)} -> {target.relative_to(ROOT)}")
        moved += 1
    print(f"Archived {moved} retired director{'y' if moved == 1 else 'ies'}.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("check", help="check release hygiene and evidence checksums")
    check.add_argument("--anonymous", action="store_true", help="also require anonymous license and package metadata")
    check.add_argument("--history", action="store_true", help="also require anonymous author/committer metadata and licenses in HEAD history")
    subparsers.add_parser("cleanup", help="archive known retired development outputs")
    args = parser.parse_args(argv)
    return check_release(args.anonymous or args.history, args.history) if args.command == "check" else archive_retired()


if __name__ == "__main__":
    raise SystemExit(main())
