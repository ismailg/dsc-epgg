import importlib.util
import sys
from pathlib import Path


_PAPER_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "neurips2026_comm"
    / "paper_data.py"
)


def _load_paper_data_module():
    spec = importlib.util.spec_from_file_location("paper_data_module", _PAPER_DATA_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_sign_flip_p_value_matches_five_seed_all_positive_case():
    paper_data = _load_paper_data_module()
    p_value = paper_data._exact_sign_flip_p_value([0.1, 0.2, 0.3, 0.4, 0.5])
    assert p_value == 0.0625


def test_gap_cell_uses_exact_sign_flip_p_value():
    paper_data = _load_paper_data_module()
    gap = paper_data.GapCell(deltas=[0.1, 0.2, 0.3, 0.4, 0.5])
    assert gap.p_value == 0.0625
    assert gap.ci_lo < gap.mean < gap.ci_hi
