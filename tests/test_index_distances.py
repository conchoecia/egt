"""Index-distance units through public distance builders and matrix assembly."""
import json
from pathlib import Path

import pandas as pd
import pytest

from egt import phylotreeumap as ptu
from egt.distance_modes import distance_metadata, check_cached_distances

SAMPLE = "Sample-123-v1"


def frame():
    return pd.DataFrame({
        "rbh": ["c", "a", "b", "d"],
        SAMPLE + "_scaf": ["chr1", "chr1", "chr1", "chr2"],
        SAMPLE + "_pos": [9000, 100, 150, 10],
        SAMPLE + "_gene": ["g3", "g1", "g2", "g4"],
        "ALG_scaf": ["A", "A", "A", "B"],
        "ALG_gene": ["c", "a", "b", "d"],
        "ALG_pos": [3, 1, 2, 4],
    })


def read_pairs(path):
    data = pd.read_csv(path, sep="\t")
    return {(r.rbh1, r.rbh2): r.distance for r in data.itertuples()}


def test_real_and_index_distances_use_distinct_units(tmp_path):
    raw, indexed = tmp_path / "real.gb.gz", tmp_path / "index.gb.gz"
    df = frame()
    ptu.rbh_to_gb(SAMPLE, df, raw)
    ptu.rbh_to_gb(SAMPLE, df, indexed, distance_mode="index")
    assert read_pairs(raw) == {("a", "b"): 50, ("a", "c"): 8900, ("b", "c"): 8850}
    assert read_pairs(indexed) == {("a", "b"): 1, ("a", "c"): 2, ("b", "c"): 1}
    assert df.equals(frame())
    metadata = json.loads(Path(str(indexed) + ".meta.json").read_text())
    assert metadata["distance_units"] == "retained_marker_steps"
    assert metadata["distance_mode"] == "index"


def test_cache_rejects_different_units_or_inputs(tmp_path):
    path = tmp_path / "example.gb.gz"
    df = frame()
    ptu.rbh_to_gb(SAMPLE, df, path, distance_mode="index")
    assert check_cached_distances(path, distance_metadata(SAMPLE, df, "index"))
    with pytest.raises(ValueError, match="distance_mode mismatch"):
        check_cached_distances(path, distance_metadata(SAMPLE, df, "coordinate"))
    df.loc[0, SAMPLE + "_pos"] = 4
    assert not check_cached_distances(path, distance_metadata(SAMPLE, df, "index"))


def test_unverified_cache_is_rebuilt(tmp_path):
    path = tmp_path / "legacy.gb.gz"
    path.write_bytes(b"legacy")
    assert not check_cached_distances(path, distance_metadata(SAMPLE, frame(), "coordinate"))
    assert not check_cached_distances(path, distance_metadata(SAMPLE, frame(), "index"))


def test_corrupt_data_cache_is_rebuilt(tmp_path):
    path = tmp_path / "example.gb.gz"
    ptu.rbh_to_gb(SAMPLE, frame(), path, distance_mode="index")
    path.write_bytes(b"interrupted output")
    assert not check_cached_distances(path, distance_metadata(SAMPLE, frame(), "index"))


def test_cli_index_defaults_use_separate_directory(monkeypatch):
    calls = []
    monkeypatch.setattr(ptu, "rbh_directory_to_distance_matrix", lambda **kw: calls.append(kw))
    ptu.main(["build-distances", "--rbh-dir", "rbhs", "--alg-name", "ALG", "--distance-mode", "index"])
    assert calls[0]["outtsv"] == "GTUMAP_index/sampledf.tsv"
    assert calls[0]["outputdir"] == "GTUMAP_index/distance_matrices/"


def test_directory_cli_and_wrapper_use_index_files(monkeypatch, tmp_path):
    rbhs = tmp_path / "rbhs"
    rbhs.mkdir()
    rbh = rbhs / f"ALG_{SAMPLE}.rbh"
    rbh.write_text("fixture")
    monkeypatch.setattr(ptu.rbh_tools, "parse_rbh", lambda _: frame())
    monkeypatch.setattr(ptu, "NCBITaxa", lambda: object())
    monkeypatch.setattr(ptu, "NCBI_taxid_to_taxdict", lambda _, taxid: {
        "taxid": taxid, "taxid_list": [], "taxid_list_str": "123", "taxname": "Example"})
    out = tmp_path / "distances"
    sampledf = tmp_path / "samples.tsv"
    assert ptu.main(["build-distances", "--rbh-dir", str(rbhs), "--alg-name", "ALG",
        "--distance-mode", "index", "--distance-dir", str(out), "--sampledf-out", str(sampledf)]) == 0
    df = pd.read_csv(sampledf, sep="\t", index_col=0)
    assert df.iloc[0].distance_mode == "index"
    assert df.iloc[0].distance_units == "retained_marker_steps"
    assert df.iloc[0].number_of_markers == 4
    assert pd.isna(df.iloc[0].genome_size)
    path = Path(df.iloc[0].dis_filepath)
    assert path.name == f"{SAMPLE}.index.gb.gz"
    assert read_pairs(path)[("a", "c")] == 2
    other = tmp_path / "wrapped.gb.gz"
    ptu.rbh_to_distance_gbgz(str(rbh), str(other), "ALG", distance_mode="index")
    assert read_pairs(other) == read_pairs(path)
    rebuilt = ptu.sampleToRbhFileDict_to_sample_matrix({SAMPLE: str(rbh)}, "ALG",
        str(out), str(tmp_path / "rebuilt.tsv"), distance_mode="index")
    assert rebuilt.iloc[0].dis_filepath == str(path)


@pytest.mark.parametrize("modes", [["index", "coordinate"], ["index", None]])
def test_combining_mixed_units_fails_before_loading_distances(modes):
    sampledf = pd.DataFrame({"sample": ["a", "b"], "dis_filepath_abs": ["no-a", "no-b"],
                             "distance_mode": modes})
    with pytest.raises(ValueError, match="mixed or unknown distance modes"):
        ptu.construct_coo_matrix_from_sampledf(sampledf, {("m1", "m2"): 0})


def test_index_matrix_keeps_exact_values(tmp_path):
    path = tmp_path / "example.gb.gz"
    ptu.rbh_to_gb(SAMPLE, frame(), path, distance_mode="index")
    sampledf = pd.DataFrame({"sample": [SAMPLE], "dis_filepath_abs": [str(path)], "distance_mode": ["index"]})
    matrix = ptu.construct_coo_matrix_from_sampledf(sampledf, {("a", "b"): 0, ("a", "c"): 1, ("b", "c"): 2, ("a", "d"): 3})
    assert matrix.toarray().tolist() == [[1, 2, 1, 0]]


@pytest.mark.parametrize("mode", ["ranks", "real", None])
def test_invalid_mode_rejected(tmp_path, mode):
    with pytest.raises(ValueError, match="distance_mode"):
        ptu.rbh_to_gb(SAMPLE, frame(), tmp_path / "bad.gb.gz", distance_mode=mode)
