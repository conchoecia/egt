"""Internal test helpers — safe to import in tests and in CI probes."""
from egt._testing.gb_roundtrip import decompose_coo_to_gbgz
from egt._testing.grid_verify_coo import grid_verify_coo

__all__ = ["decompose_coo_to_gbgz", "grid_verify_coo"]
