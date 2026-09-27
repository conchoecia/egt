"""Distance units and safeguards for reusing generated distance files."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd

MODES = ("coordinate", "index")


def validate_mode(mode):
    if mode not in MODES:
        raise ValueError(f"distance_mode must be one of {MODES}, got {mode!r}")
    return mode


def distance_filename(sample, mode):
    validate_mode(mode)
    return f"{sample}{'.index' if mode == 'index' else ''}.gb.gz"


def distance_metadata(sample, frame, mode):
    validate_mode(mode)
    cols = ["rbh", f"{sample}_scaf", f"{sample}_pos"]
    values = pd.util.hash_pandas_object(frame[cols], index=False).to_numpy().tobytes()
    return {
        "schema_version": 1,
        "sample": sample,
        "distance_mode": mode,
        "distance_units": "retained_marker_steps" if mode == "index" else "input_coordinate_units",
        "ranking_scope": "retained markers within each chromosome" if mode == "index" else None,
        "coordinate_ties": "lexicographic marker ID" if mode == "index" else None,
        "input_rows": len(frame),
        "input_sha256": hashlib.sha256(values).hexdigest(),
    }


def metadata_path(outfile):
    return Path(str(outfile) + ".meta.json")


def write_metadata(outfile, metadata):
    metadata = dict(metadata, output_sha256=file_digest(outfile))
    target = metadata_path(outfile)
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, prefix=target.name + ".", delete=False) as tmp:
        json.dump(metadata, tmp, indent=2)
        tmp.write("\n")
    os.replace(tmp.name, target)


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_cached_distances(outfile, expected):
    """Rebuild unverified/stale caches; reject known unit mismatches."""
    if not Path(outfile).exists():
        return False
    sidecar = metadata_path(outfile)
    if not sidecar.exists():
        return False
    found = json.loads(sidecar.read_text())
    for field in ("sample", "distance_mode"):
        if found.get(field) != expected[field]:
            raise ValueError(f"Cached distance {field} mismatch for {outfile}. Use a fresh output path.")
    return found.get("input_sha256") == expected["input_sha256"] and found.get("output_sha256") == file_digest(outfile)
