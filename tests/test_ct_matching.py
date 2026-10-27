"""Clinical-trial MeSH and free-text matching must not link on partial terms."""
import logging
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.CRITICAL)
spec = spec_from_file_location('ct02', ROOT / 'PYTHON' / '04-02-heim-ct-map-diseases.py')
m = module_from_spec(spec); spec.loader.exec_module(m)
MP = m.GBD_CONDITION_MAPPING

@pytest.mark.parametrize('mesh,cause', [
    ('Neoplasms', 'Breast cancer'), ('Infant, Newborn', 'Neonatal disorders'), ('Pregnancy', 'Maternal disorders'),
    ('Child', 'Interpersonal violence'),
])
def test_mesh_partial_term_does_not_match(mesh, cause):
    assert cause in MP
    assert cause not in m.match_mesh_term(mesh, MP)

@pytest.mark.parametrize('mesh,cause', [('Malaria', 'Malaria'), ('Tuberculosis, Pulmonary', 'Tuberculosis')])
def test_mesh_genuine_matches(mesh, cause):
    assert cause in MP and cause in m.match_mesh_term(mesh, MP)

def test_keyword_cancer_does_not_capture_every_cancer_trial():
    assert 'Other malignant neoplasms' in MP
    causes = m.finalise_causes(set(m.match_condition_text('Breast Cancer', MP)))
    assert 'Other malignant neoplasms' not in causes and 'Breast cancer' in causes

def test_keyword_whole_word():
    # 'stroke' must not fire inside 'heatstroke'; genuine 'ischemic stroke' must
    assert 'Stroke' not in m.match_condition_text('Heatstroke', MP)
    assert 'Stroke' in m.match_condition_text('Acute Ischemic Stroke', MP)

def test_mesh_qualifier_form_matches_curated_term():
    # 'Malaria, Vivax' is not itself a curated term; it must match 'Malaria' through the qualifier rule
    assert 'Malaria, Vivax' not in [t for t in MP['Malaria']['mesh_terms']]
    assert 'Malaria' in m.match_mesh_term('Malaria, Vivax', MP)

def test_residual_cancer_category_removed_when_specific_cancer_matched_via_mesh():
    causes = m.finalise_causes({'Other malignant neoplasms', 'Breast cancer'})
    assert causes == {'Breast cancer'}
    assert m.finalise_causes({'Other malignant neoplasms'}) == {'Other malignant neoplasms'}

def test_trachoma_trials_do_not_come_from_genital_chlamydia():
    assert 'Trachoma' not in m.match_mesh_term('Chlamydia trachomatis', MP)


def test_registry_dalys_join_is_exact_not_substring():
    gbd = {'Hodgkin lymphoma': 7.8e6, 'Non-Hodgkin lymphoma': 1.2e6, 'Stroke': 1.6e8}
    assert m.lookup_dalys('Non-Hodgkin lymphoma', gbd) == 1.2e6
    assert m.lookup_dalys('Hodgkin lymphoma', gbd) == 7.8e6
    assert m.lookup_dalys('Tracheal, bronchus, and lung cancer', gbd) == 0

def test_parallel_worker_applies_residual_rule():
    m._MESH_BY_NCT = {'NCT1': ['Neoplasms', 'Breast Neoplasms']}; m._CONDITIONS_BY_NCT = {'NCT1': ['Breast Cancer']}
    nct, causes = m.map_trial_fast('NCT1')
    assert 'Breast cancer' in causes and 'Other malignant neoplasms' not in causes
