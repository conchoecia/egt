"""`egt go benchmark --ref {goatools,camps,simakov,alg}` dispatch wrapper.

Selects one of the egt.go.benchmarks submodules at runtime. Remaining
argv is forwarded to that submodule's own argparse.
"""
from __future__ import annotations

import importlib
import sys


REFS: dict[str, str] = {
    "goatools": "egt.go.benchmarks.goatools_ref",
    "camps":    "egt.go.benchmarks.camps",
    "simakov":  "egt.go.benchmarks.simakov",
    "alg":      "egt.go.benchmarks.alg_structure",
}


def _print_dispatch_help() -> None:
    print("usage: egt go benchmark --ref {" + ",".join(sorted(REFS)) + "} [args…]\n")
    print("Cross-check defining-pair catalog against external references.")
    print("Pass --help AFTER --ref to see the selected reference's options.")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else []
    # No args or bare --help → show dispatch help.
    if not argv or argv[0] in ("-h", "--help"):
        _print_dispatch_help()
        return 0
    # Pull --ref and dispatch. Accept either `--ref X rest…` or `--ref=X rest…`.
    ref = None
    rest: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ref":
            if i + 1 >= len(argv):
                print("egt go benchmark: --ref requires a value", file=sys.stderr)
                return 2
            ref = argv[i + 1]
            i += 2
            continue
        if a.startswith("--ref="):
            ref = a.split("=", 1)[1]
            i += 1
            continue
        rest.append(a)
        i += 1
    if ref is None:
        print("egt go benchmark: --ref is required", file=sys.stderr)
        return 2
    if ref not in REFS:
        print(f"egt go benchmark: unknown --ref {ref!r}; "
              f"choose from {sorted(REFS)}", file=sys.stderr)
        raise SystemExit(2)
    mod = importlib.import_module(REFS[ref])
    fn = getattr(mod, "main")
    rv = fn(rest)
    return int(rv) if rv is not None else 0
