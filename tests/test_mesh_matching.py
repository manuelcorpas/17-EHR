"""Publication MeSH terms must map to a GBD cause only on a genuine term match."""
import logging
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
logging.disable(logging.CRITICAL)
spec = spec_from_file_location('bhem01', ROOT / 'PYTHON' / '03-01-bhem-map-diseases.py')
m = module_from_spec(spec); spec.loader.exec_module(m)
MAP = m.GBD_MESH_MAPPING
mapped = lambda mesh: set(m.map_publication_to_causes(mesh, MAP))

@pytest.mark.parametrize('mesh,cause', [
    ('Sleep; Genome-Wide Association Study', 'African trypanosomiasis'),   # 'sleep' inside 'sleeping sickness'
    ('Liver; Alanine Transaminase', 'Food-borne trematodiases'),           # 'liver' inside 'liver flukes'
    ('Disease; Polymorphism, Single Nucleotide', 'Chagas disease'),        # 'disease' inside 'chagas disease'
])
def test_short_article_term_inside_cause_term_does_not_match(mesh, cause):
    assert cause in MAP, f'{cause} not a mapping key'  # guard: a misspelt key would pass vacuously
    assert cause not in mapped(mesh)

@pytest.mark.parametrize('mesh,cause', [
    ('Malaria', 'Malaria'),
    ('Malaria, Falciparum', 'Malaria'),          # MeSH subheading of a cause term
    ('Tuberculosis, Pulmonary', 'Tuberculosis'),
    ('Trypanosomiasis, African', 'African trypanosomiasis'),
])
def test_genuine_matches_still_match(mesh, cause):
    assert cause in MAP, f'{cause} not a mapping key'
    assert cause in mapped(mesh)

def test_organism_heading_for_genital_chlamydia_does_not_map_to_trachoma():
    assert 'Trachoma' in MAP
    assert 'Trachoma' not in mapped('Chlamydia trachomatis; Serologic Tests')
    assert 'Trachoma' in mapped('Trachoma')
