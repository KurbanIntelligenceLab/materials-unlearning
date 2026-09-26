#!/usr/bin/env python3
"""Build every project module and reject unapproved transitive theorem axioms."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}

def run(args):
    result = subprocess.run(args, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
    return result.stdout

def main():
    lake = os.environ.get("LAKE") or shutil.which("lake")
    if not lake:
        raise SystemExit("lake not found; install elan or set LAKE to its absolute path")
    names = []
    for source in sorted((ROOT / "DeletionFloor").glob("*.lean")):
        content = source.read_text()
        if re.search(r"\b(?:sorry|admit|sorryAx|axiom|unsafe|native_decide)\b", content):
            raise SystemExit(f"Forbidden proof token in {source.name}")
        if re.findall(r"^namespace (\w+)", content, re.M) != ["DeletionFloor"]:
            raise SystemExit(f"Unexpected namespace layout in {source.name}")
        names.extend("DeletionFloor." + name for name in re.findall(r"^\s*theorem (\w+)", content, re.M))
    if not names or len(set(names)) != len(names):
        raise SystemExit("Missing or duplicate theorem declarations")
    run([lake, "build"])
    fd, filename = tempfile.mkstemp(prefix="AxiomAudit", suffix=".lean", dir=ROOT)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write("import DeletionFloor\n" + "\n".join("#print axioms " + n for n in names) + "\n")
        output = run([lake, "env", "lean", filename])
    finally:
        Path(filename).unlink(missing_ok=True)
    audited = set()
    for name, axioms in re.findall(r"'([^']+)' depends on axioms: \[([^\]]*)\]", output, re.S):
        dependencies = {a.strip() for a in axioms.split(",") if a.strip()}
        if dependencies - ALLOWED:
            raise SystemExit(f"Unapproved axioms for {name}: {sorted(dependencies - ALLOWED)}")
        audited.add(name)
    audited.update(re.findall(r"'([^']+)' does not depend on any axioms", output))
    if audited != set(names):
        raise SystemExit(f"Audit coverage mismatch: missing={sorted(set(names)-audited)} extra={sorted(audited-set(names))}")
    print(f"PASS: {len(names)} project theorems built and audited; only standard mathematical axioms.")

if __name__ == "__main__":
    main()
