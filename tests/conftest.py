# tests/conftest.py

import pytest
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

@pytest.fixture
def mock_project_dir(tmp_path: Path) -> Path:
    """Creates a temporary project directory with a mock config and dataset.

    Args:
        tmp_path (Path): Built-in pytest fixture for temporary directories.

    Returns:
        Path: The absolute path to the generated config.yaml file.
    """
    data_dir: Path = tmp_path / "data"
    data_dir.mkdir()
    
    # 1. Create a tiny mock CSV dataset
    mock_data = pd.DataFrame({
        't0': [100.0, 500.0, 1000.0],
        'a': [0.1, 0.15, 0.2],
        'm1': [0.6, 1.0, 1.2],
        'm2': [0.6, 0.8, 1.0]
    })
    pop_path: Path = data_dir / "mock_population.csv"
    mock_data.to_csv(pop_path, index=False)
    
    # 2. Create a mock configuration
    config: Dict = {
        'global': {
            'frequency': {
                'f_min': 1.0e-4,
                'f_max': 1.0e-2,
                'N_fbins': 10,
                'f_scale': 'log10'
            },
            'save_results': True, # We want to test the saving functionality!
            'save_directory': './results/' 
        },
        'cosmology': {
            'standard': True,
            'standard_cosmology': 'Planck18',
            'z_min': 0.0,
            'z_max': 5.0,
            'N_zbins': 5,
            'z_scale': 'linear'
        },
        'population': {
            'population_name': 'Test_Catalogue',
            'population_path': './data/mock_population.csv',
            'total_population_mass': 1.0e10
        },
        'SFR': {
            'SFR_name': 'madau_and_dickinson'
        }
    }
    
    config_path: Path = tmp_path / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return config_path