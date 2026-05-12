from __future__ import annotations

import numpy as np
import pytest

from egt.bokeh_helper import convert_hex_string_to_colorvalues, remove_ticks


class Axis:
    def __init__(self):
        self.major_tick_line_color = "black"
        self.minor_tick_line_color = "black"
        self.major_label_text_font_size = "10pt"


class DummyPlot:
    def __init__(self):
        self.xaxis = Axis()
        self.yaxis = Axis()


def test_convert_hex_string_to_colorvalues_appends_alpha():
    arr = convert_hex_string_to_colorvalues(["#1F779A", "#9F0000"])
    assert np.array_equal(arr, np.array([0x1F779AFF, 0x9F0000FF], dtype=np.uint32))


def test_convert_hex_string_to_colorvalues_requires_hash_prefix():
    with pytest.raises(ValueError, match="does not start with a #"):
        convert_hex_string_to_colorvalues(["1F779A"])


def test_remove_ticks_clears_tick_lines_and_labels():
    plot = DummyPlot()
    remove_ticks(plot)
    assert plot.xaxis.major_tick_line_color is None
    assert plot.xaxis.minor_tick_line_color is None
    assert plot.yaxis.major_tick_line_color is None
    assert plot.yaxis.minor_tick_line_color is None
    assert plot.xaxis.major_label_text_font_size == "0pt"
    assert plot.yaxis.major_label_text_font_size == "0pt"

