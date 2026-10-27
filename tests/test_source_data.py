from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('sd', ROOT / 'PYTHON' / '08-02-source-data.py'); sd = module_from_spec(spec); spec.loader.exec_module(sd)

def test_sheets_cover_every_display_item_with_expected_sizes():
    sh = sd.sheets()
    for name in ['Fig2a_country_EAS', 'Fig2b_biobank_EAS', 'Fig2c_publication_share', 'Fig3a_trial_sites', 'Fig3b_GS_intensity',
                 'Fig4b_e_semantic', 'Fig5_unified_top30', 'Table1_top10', 'EDFig1a', 'EDFig1b', 'EDFig1c', 'EDFig2_sensitivity',
                 'EDFig3_regional', 'SuppTable_SX_scores', 'SuppTable_SY_counts']:
        assert name in sh and len(sh[name]) > 0, name
    assert len(sh['Fig2b_biobank_EAS']) == 70 and len(sh['Fig5_unified_top30']) == 30 and len(sh['Table1_top10']) == 10
    assert len(sh['EDFig1b']) == 26 and len(sh['Fig2c_publication_share']) == 2 and len(sh['SuppTable_SY_counts']) == 179
    assert sh['Table1_top10']['rank'].tolist() == list(range(1, 11))
