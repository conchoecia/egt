"""Top-level CLI dispatch tests for `egt go …`."""
from __future__ import annotations

import importlib

import pytest

from egt.cli import SUBCOMMANDS, _dispatch, _print_help
from egt.go_subcommands import GO_SUBCOMMANDS


def test_go_is_registered_as_umbrella():
    assert "go" in SUBCOMMANDS
    entry = SUBCOMMANDS["go"]
    assert isinstance(entry, tuple)
    sub, hlp = entry
    assert sub is GO_SUBCOMMANDS
    assert "GO enrichment" in hlp


@pytest.mark.parametrize("name", sorted(GO_SUBCOMMANDS))
def test_every_go_subcommand_has_main_argv(name):
    module_path, help_text = GO_SUBCOMMANDS[name]
    assert help_text, f"{name} help should be non-empty"
    mod = importlib.import_module(module_path)
    assert hasattr(mod, "main"), f"{module_path} missing main()"
    # main() must be callable with an argv list — verify via --help.
    # Most sub-modules use argparse which SystemExits on --help; the
    # benchmark dispatcher prints its own dispatch help and returns 0.
    if name == "benchmark":
        rc = mod.main(["--help"])
        assert rc == 0
    else:
        with pytest.raises(SystemExit) as excinfo:
            mod.main(["--help"])
        assert excinfo.value.code == 0


def test_dispatch_empty_argv_prints_help(capsys):
    rc = _dispatch(SUBCOMMANDS, [], "egt")
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage: egt" in out


def test_dispatch_dash_h_prints_help(capsys):
    rc = _dispatch(SUBCOMMANDS, ["-h"], "egt")
    assert rc == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_dispatch_unknown_returns_2(capsys):
    rc = _dispatch(SUBCOMMANDS, ["not-a-real-command"], "egt")
    assert rc == 2
    err = capsys.readouterr().err
    assert "unknown command" in err


def test_dispatch_nested_go_group(capsys):
    rc = _dispatch(SUBCOMMANDS, ["go"], "egt")
    assert rc == 0
    out = capsys.readouterr().out
    # The go subgroup's own help is printed; it lists go sub-subcommands.
    assert "sweep" in out or "usage" in out


def test_dispatch_nested_go_sweep_returns_0_on_help(capsys):
    # `egt go sweep --help` dispatches to sweep.main which argparse exits 0 on.
    with pytest.raises(SystemExit) as excinfo:
        _dispatch(SUBCOMMANDS, ["go", "sweep", "--help"], "egt")
    assert excinfo.value.code == 0


def test_dispatch_main_entry_matches():
    from egt import cli as climod
    # main() wraps _dispatch — ensure --help returns 0.
    rc = climod.main(["--help"])
    assert rc == 0


def test_print_help_covers_both_tuple_and_dict_entries(capsys):
    # Verify _print_help surfaces both leaf ("sweep") and group ("go") rows.
    _print_help(SUBCOMMANDS, "egt")
    out = capsys.readouterr().out
    assert "go" in out
    assert "defining-features" in out
