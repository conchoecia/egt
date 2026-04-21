from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from egt import fourier_of_rates as fr


def test_parse_args_and_signal_helpers(tmp_path: Path):
    rates = tmp_path / "rates.tsv"
    pd.DataFrame(
        {
            "age": [-4.0, -3.0, -2.0, -1.0],
            "fusion_rate_at_this_age_mean": [1.0, 0.0, 1.0, 0.0],
        }
    ).to_csv(rates, sep="\t", index=False)

    args = fr.parse_args(
        [
            "--rates",
            str(rates),
            "--custom_peaks",
            "20, 40",
            "--include_unpadded",
        ]
    )
    assert args.custom_peaks == [20.0, 40.0]
    assert args.include_unpadded is True

    time = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, 0.0, -1.0])
    spectrum, frequencies, yf = fr.fft_unpadded(time, values)
    padded_spectrum, padded_freqs, padded_yf = fr.fft_padded(time, values, zero_padding_factor=4)
    assert len(spectrum) == len(frequencies) == len(yf[: len(spectrum)])
    assert len(padded_spectrum) == len(padded_freqs) == len(padded_yf[: len(padded_spectrum)])

    local_peaks, local_freqs = fr.find_spectrum_peaks(np.array([0.0, 2.0, 0.5, 1.5, 0.1]), np.arange(5), n_peaks=2)
    sorted_peaks, sorted_freqs = fr.find_spectrum_peaks(
        np.array([0.0, 1.0, 3.0, 2.0]), np.arange(4), n_peaks=2, use_peak_detection=False
    )
    assert set(local_peaks) == {1, 3}
    assert set(local_freqs) == {1, 3}
    assert list(sorted_peaks) == [2, 3]
    assert list(sorted_freqs) == [2, 3]


def test_background_alignment_and_cache_loader(tmp_path: Path):
    magnitudes = np.array([1.0, 2.0, 3.0])
    heights = np.array([2.0, 2.0, 2.0])
    probs = fr.exponential_background(magnitudes, heights)
    assert np.all(probs >= 0)
    assert probs[-1] > probs[0]

    time = np.linspace(0.0, 9.0, 10)
    residuals = np.sin(2 * np.pi * time / 5.0)
    fitted = fr.fit_sine_wave_to_residuals(time, residuals, period=5.0)
    corr = np.corrcoef(residuals, fitted)[0, 1]
    assert corr > 0.8

    outprefix = tmp_path / "cached"
    cached_path = Path(str(outprefix) + "_chunk_support.tsv")
    pd.DataFrame({"chunks": [21], "period_20.00_observed": [0.9]}).to_csv(cached_path, sep="\t", index=False)
    cached = fr.load_cached_simulation_data(str(outprefix))
    assert list(cached.columns) == ["chunks", "period_20.00_observed"]
    assert fr.load_cached_simulation_data(str(tmp_path / "missing")) is None


def test_fourier_of_time_series_dispatches_and_validates_spacing(monkeypatch, tmp_path: Path):
    df = pd.DataFrame({"age": [-4.0, -3.0, -2.0, -1.0], "rate": [0.0, 1.0, 0.0, -1.0]})
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        fr,
        "run_single_analysis",
        lambda time, values, polynomial, outprefix, spectrum_type, custom_peaks=None: calls.append(
            (outprefix, spectrum_type)
        ),
    )

    fr.fourier_of_time_series(df, "age", "rate", 2, str(tmp_path / "fft"), custom_peaks=[0.05], include_unpadded=True)
    assert calls == [
        (str(tmp_path / "fft_padded"), "padded"),
        (str(tmp_path / "fft_unpadded"), "unpadded"),
    ]

    uneven = pd.DataFrame({"age": [-4.0, -2.0, -1.0], "rate": [0.0, 1.0, 0.0]})
    with pytest.raises(ValueError, match="not evenly spaced"):
        fr.fourier_of_time_series(uneven, "age", "rate", 2, str(tmp_path / "bad"), custom_peaks=[0.05])


def test_run_single_analysis_with_cached_simulation_data(monkeypatch, tmp_path: Path):
    chunk_support = pd.DataFrame(
        {
            "chunks": [10, 21, 30],
            "period_20.00_observed": [0.1, 0.92, 0.96],
            "period_20.00_expected": [0.2, 0.3, 0.4],
        }
    )
    monkeypatch.setattr(fr, "load_cached_simulation_data", lambda _prefix: chunk_support)

    time = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    values = np.array([0.0, 2.0, 0.0, -2.0, 0.0, 2.0])
    outprefix = str(tmp_path / "analysis")
    fr.run_single_analysis(time, values, polynomial=2, outprefix=outprefix, spectrum_type="padded", custom_peaks=[0.05])
    assert (tmp_path / "analysis.pdf").exists()


def test_main_filters_time_range_and_empty_result(monkeypatch, tmp_path: Path):
    rates = tmp_path / "rates.tsv"
    pd.DataFrame(
        {
            "age": [-10.0, -5.0, -2.0],
            "fusion_rate_at_this_age_mean": [1.0, 2.0, 3.0],
        }
    ).to_csv(rates, sep="\t", index=False)

    called: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(
        fr,
        "fourier_of_time_series",
        lambda df, *args, **kwargs: called.setdefault("df", df.copy()),
    )

    assert fr.main(["--rates", str(rates), "--min_time", "1", "--max_time", "6"]) == 0
    filtered = called["df"]
    assert filtered["age"].tolist() == [-5.0, -2.0]

    with pytest.raises(SystemExit, match="1"):
        fr.main(["--rates", str(rates), "--min_time", "1000", "--max_time", "1001"])


def test_run_single_analysis_without_cache_uses_simulation_results(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fr, "load_cached_simulation_data", lambda _prefix: None)
    monkeypatch.setattr(
        fr,
        "r_simulations",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "chunks": [21, 30],
                "period_20.00_observed": [0.92, 0.97],
                "period_20.00_expected": [0.3, 0.4],
            }
        ),
    )

    time = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    values = np.array([0.0, 2.0, 0.0, -2.0, 0.0, 2.0])
    outprefix = str(tmp_path / "significant")
    fr.run_single_analysis(time, values, polynomial=2, outprefix=outprefix, spectrum_type="padded", custom_peaks=[0.05])
    assert (tmp_path / "significant.pdf").exists()


def test_fit_sine_wave_fallback_and_main_column_validation(monkeypatch, tmp_path: Path):
    time = np.array([0.0, 1.0, 2.0, 3.0])
    residuals = np.array([1.0, -1.0, 1.0, -1.0])
    monkeypatch.setattr(fr, "correlate", lambda *_args, **_kwargs: np.array([1.0]))
    fitted = fr.fit_sine_wave_to_residuals(time, residuals, period=2.0)
    assert len(fitted) == len(time)

    rates = tmp_path / "rates.tsv"
    pd.DataFrame({"age": [-2.0, -1.0], "value": [1.0, 2.0]}).to_csv(rates, sep="\t", index=False)
    with pytest.raises(ValueError, match="not in the dataframe"):
        fr.main(["--rates", str(rates), "--ratecol", "missing_col"])

    with pytest.raises(ValueError, match="not in the dataframe"):
        fr.main(["--rates", str(rates), "--agecol", "missing_age", "--ratecol", "value"])


def test_r_simulations_writes_cache_and_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(fr.np.random, "permutation", lambda n: np.arange(n))
    monkeypatch.setattr(fr, "exponential_background", lambda magnitude, heights: np.linspace(0.1, 0.9, len(magnitude)))

    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.0, 1.0, 0.0, 1.0, 0.0])
    magnitudes = np.array([0.5, 0.8, 0.3])
    peaks = [0.5, 0.25]
    outprefix = str(tmp_path / "rsim")

    summary = fr.r_simulations(
        time,
        values,
        peaks_to_test=peaks,
        magnitudes=magnitudes,
        polynomial=2,
        outprefix=outprefix,
        padded=False,
        simulations=1,
    )

    assert list(summary.columns) == [
        "chunks",
        "simulations",
        "outprefix",
        "period_2.00_observed",
        "period_2.00_expected",
        "period_4.00_observed",
        "period_4.00_expected",
    ]
    assert Path(outprefix + "_rsims.pdf").exists()
    assert Path(outprefix + "_rsims.tsv").exists()
    assert Path(outprefix + "_chunk_support.tsv").exists()
