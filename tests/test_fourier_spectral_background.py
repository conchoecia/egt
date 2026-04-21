from __future__ import annotations

import runpy
from pathlib import Path

import matplotlib.pyplot as plt


def test_script_runs_and_saves_pdf(monkeypatch):
    saved = []
    monkeypatch.setattr(plt, "savefig", lambda path: saved.append(path))

    script = Path(__file__).resolve().parents[1] / "src" / "egt" / "fourier_spectral_background.py"
    runpy.run_path(str(script), run_name="__main__")

    assert saved == ["spectral_background.pdf"]
