from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, save_npz

from egt import phylotreeumap_testpixels as ptp


def test_log_scale_and_resize_image():
    assert ptp.log_scale(0, 10) == 0

    image = ptp.Image.new("RGB", (16, 16), color=(255, 255, 255))
    resized = ptp.resize_image(image, 4)
    assert resized.size == (2, 2)

    with pytest.raises(IOError, match="even number"):
        ptp.resize_image(image, 3)


def test_plot_coo_matrix_writes_pngs(tmp_path: Path, monkeypatch):
    matrix = coo_matrix(np.array([[1, 2, 0], [0, 3, 4], [5, 0, 6]], dtype=float))
    coopath = tmp_path / "matrix.npz"
    save_npz(coopath, matrix)

    cwd = Path.cwd()
    try:
        import os
        os.chdir(tmp_path)
        ptp.plot_coo_matrix(str(coopath))
    finally:
        os.chdir(cwd)

    assert (tmp_path / "coo_matrix_rowUn_colUn.png").exists()
    assert (tmp_path / "coo_matrix_rowUn_colSort.png").exists()
    assert (tmp_path / "coo_matrix_rowSort_colUn.png").exists()
    assert (tmp_path / "coo_matrix_rowSort_colSort.png").exists()


def test_main_dispatches(monkeypatch):
    seen = {}
    monkeypatch.setattr(ptp, "plot_coo_matrix", lambda path: seen.setdefault("path", path))
    assert ptp.main() == 0
    assert seen["path"] == "allsamples.coo.npz"
