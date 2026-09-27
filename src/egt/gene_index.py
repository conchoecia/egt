"""Gene-order coordinates among the retained markers on each chromosome."""
import numpy as np
import pandas as pd


def index_coordinates(sample: str, rbhdf: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose observed positions are consecutive 1-based ranks.

    Rows without a scaffold or coordinate are excluded. Retained marker IDs
    must be unique. Coordinates sort numerically; ties break by marker ID,
    a reproducible convention that does not imply known biological order.
    Ranks count input markers, not genes absent from this dataframe.
    Original row order, index labels and auxiliary columns are preserved.
    """
    scaffold, position = f"{sample}_scaf", f"{sample}_pos"
    if not isinstance(rbhdf, pd.DataFrame) or not rbhdf.columns.is_unique:
        raise ValueError("Supply a dataframe with unique column names")
    required = {"rbh", scaffold, position}
    if not required.issubset(rbhdf.columns):
        raise ValueError(f"Missing index-coordinate columns: {sorted(required - set(rbhdf.columns))}")
    keep = rbhdf[scaffold].notna() & rbhdf[position].notna()
    result = rbhdf.iloc[np.flatnonzero(keep.to_numpy())].copy()
    for column in ("rbh", scaffold):
        if not all(isinstance(v, str) and v.strip() for v in result[column]):
            raise ValueError(f"Retained {column} values must be nonempty strings")
    if result["rbh"].duplicated().any():
        raise ValueError("Retained marker IDs must be unique within a sample")
    if any(isinstance(v, (bool, np.bool_, complex, np.complexfloating)) for v in result[position]):
        raise ValueError("Index coordinates must be finite real numbers, not booleans or complex values")
    try:
        numeric = pd.to_numeric(result[position], errors="raise")
        finite = np.isfinite(numeric.to_numpy(dtype=np.float64))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Index coordinates must be finite real numbers") from error
    if not finite.all():
        raise ValueError("Index coordinates must be finite real numbers")
    ordered = pd.DataFrame({"scaffold": result[scaffold].to_numpy(),
                            "position": numeric.to_numpy(), "marker": result["rbh"].to_numpy()})
    ordered = ordered.sort_values(["scaffold", "position", "marker"], kind="stable")
    ranks = np.empty(len(result), dtype=np.int64)
    ranks[ordered.index.to_numpy()] = ordered.groupby("scaffold", sort=False).cumcount().to_numpy() + 1
    result[position] = ranks
    return result
