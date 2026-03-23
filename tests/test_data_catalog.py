from src.analysis.build_data_catalog import build_catalog


def test_catalog_contains_phase3_train_and_eval():
    rows = build_catalog("/Users/mbp17/POSTDOC/NPS26/dsc-epgg")
    roots = {str(r["root"]) for r in rows}
    assert "outputs/train/phase3_annealed_trimmed_15seeds" in roots
    assert "outputs/eval/phase3_annealed_ext150k_15seeds" in roots
