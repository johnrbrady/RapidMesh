"""Clean-checkout smoke gate.

Catches two failure classes that unit tests do not:
  1. a tool that will not run from its documented location;
  2. a module in `tools/` whose name shadows the standard library.

The second is not hypothetical. `tools/inspect.py` shadowed stdlib `inspect`
and broke every script run from `tools/`, because running a script puts its
directory on sys.path[0]. Modules inside `src/rapidmesh/` are namespaced by the
package and are not at risk, so only `tools/` is checked.

    python tools/smoke.py
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def check_shadowing() -> list[str]:
    stdlib = set(sys.stdlib_module_names)
    return sorted(p.name for p in (ROOT / "tools").glob("*.py") if p.stem in stdlib)


def check_runs() -> list[str]:
    failures = []
    checks = [
        ([sys.executable, "-c", "from rapidmesh import cli; cli.main(['--help'])"], "CLI"),
        ([sys.executable, str(ROOT / "tools" / "bench_synthetic.py"), "--help"], "bench_synthetic"),
        ([sys.executable, str(ROOT / "tools" / "e57_inventory.py")], "e57_inventory"),
        ([sys.executable, str(ROOT / "tools" / "check_laz_precision.py")], "check_laz_precision"),
        ([sys.executable, str(ROOT / "tools" / "measure_peak_memory.py"), "--help"], "measure_peak_memory"),
    ]
    for cmd, name in checks:
        r = subprocess.run(cmd, capture_output=True, cwd=ROOT, text=True)
        # A tool invoked with no arguments may exit non-zero; an ImportError may not.
        blob = r.stdout + r.stderr
        if "ImportError" in blob or "ModuleNotFoundError" in blob or "Traceback" in blob and "usage" not in blob.lower():
            failures.append(f"{name}: {blob.strip().splitlines()[-1] if blob.strip() else 'no output'}")
    return failures


def main() -> int:
    ok = True
    shadowed = check_shadowing()
    if shadowed:
        print(f"FAIL  tools/ modules shadowing the standard library: {shadowed}")
        ok = False
    else:
        print("ok    no stdlib shadowing in tools/")

    failures = check_runs()
    if failures:
        print("FAIL  tools that do not run from a clean checkout:")
        for f in failures:
            print(f"        {f}")
        ok = False
    else:
        print("ok    CLI, bench, inventory, LAZ and peak-memory tools all import and run")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
