"""Every GBD Level 2 label in the registry must map explicitly; no silent 'Other NCD' fallback."""
import json
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location('bhem', ROOT / 'PYTHON' / '03-03-bhem-compute-metrics.py')
bhem = module_from_spec(spec); spec.loader.exec_module(bhem)

def registry():
    r = json.load(open(ROOT / 'DATA' / 'gbd_disease_registry.json'))
    return r.get('diseases', r)

def test_every_registry_label_is_mapped():
    labels = {v.get('gbd_level2') for v in registry().values()}
    for label in labels:
        bhem.map_gbd_category(label)  # raises on unknown

@pytest.mark.parametrize('disease,expected', [
    ('Malaria', 'Neglected'), ('Leprosy', 'Neglected'), ('Ebola', 'Neglected'),
    ('Zika virus', 'Neglected'), ('Measles', 'Infectious'), ('Meningitis', 'Infectious'),
    ('Self-harm', 'Injuries'), ('Breast cancer', 'Cancer'),
])
def test_known_diseases(disease, expected):
    assert bhem.map_gbd_category(registry()[disease]['gbd_level2']) == expected

def test_unknown_label_raises():
    with pytest.raises(KeyError):
        bhem.map_gbd_category('Not a GBD category')

def test_global_south_flag_applies_to_registry_infectious_label():
    assert 'Other infectious' in bhem.GLOBAL_SOUTH_GBD_CATEGORIES
