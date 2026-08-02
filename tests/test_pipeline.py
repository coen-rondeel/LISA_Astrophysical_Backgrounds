"""End-to-end tests for the GWB pipeline."""

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import yaml

from lisaastrophysicalbackgrounds.gravitational_wave_background import (
    GravitationalWaveBackground,
)
from lisaastrophysicalbackgrounds.utils import get_config


# We use the mock_project_dir fixture defined in conftest.py
def test_initialization_and_grids(mock_project_dir: Path) -> None:
    """Tests if the class initializes properly and grids are built correctly."""
    gwb = GravitationalWaveBackground(str(mock_project_dir))

    N_fbins = gwb.config["global"]["frequency"]["N_fbins"]
    assert len(gwb.f_vals) == N_fbins
    assert len(gwb.f_bins) == N_fbins + 1

    N_zbins = gwb.config["cosmology"]["N_zbins"]
    assert len(gwb.cosmology.z_vals) == N_zbins
    assert len(gwb.cosmology.z_bins) == N_zbins + 1

    assert len(gwb.population.M_ch) > 0


# Since 'get_GWB_plot' currently raises NotImplementedError, we mock (bypass) it
# just to ensure the rest of calculate_GWB() and save_results() finish successfully.
@patch("lisaastrophysicalbackgrounds.gravitational_wave_background.get_GWB_plot")
def test_full_calculation_and_saving(mock_plot, mock_project_dir: Path) -> None:
    """Tests the main calculation loop and that files are saved to the correct paths."""
    gwb = GravitationalWaveBackground(str(mock_project_dir))

    gwb.calculate_GWB()

    assert gwb.omega_f.shape == gwb.f_vals.shape
    assert gwb.N_sources_f.shape == gwb.f_vals.shape

    save_dir = Path(gwb.config["global"]["save_directory"])
    expected_file = save_dir / "Test_Catalogue_gwb_results.h5"

    assert expected_file.exists(), f"Expected file {expected_file} was not created!"

    expected_plot_path = (
        save_dir / "GWB_for_Test_Catalogue_with_madau_and_dickinson.png"
    )
    mock_plot.assert_called_once()
    assert str(expected_plot_path) in str(mock_plot.call_args[1]["save_path"])

    # the IMF frames must be recoverable from the output file
    with h5py.File(expected_file, "r") as hf:
        assert hf.attrs["SFH_reference_IMF"] == "salpeter"
        assert hf.attrs["population_IMF"] == "kroupa"
        assert hf.attrs["IMF_correction_factor"] == pytest.approx(0.66)


@patch("lisaastrophysicalbackgrounds.gravitational_wave_background.get_GWB_plot")
def test_omega_scales_with_imf_factor(mock_plot, mock_project_dir: Path) -> None:
    """Omega scales linearly with the applied IMF correction.

    The fixture pairs a Salpeter SFRD (madau_and_dickinson) with a Kroupa
    population, so the default run already carries the 0.66 conversion.
    """
    corrected = GravitationalWaveBackground(str(mock_project_dir))
    assert corrected.SFH.imf_factor == pytest.approx(0.66)
    corrected.calculate_GWB()

    config = get_config(str(mock_project_dir))
    config["SFH"]["SFH_IMF_correction"] = 1.0
    with open(mock_project_dir, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    native = GravitationalWaveBackground(str(mock_project_dir))
    assert native.SFH.imf_factor == 1.0
    native.calculate_GWB()

    np.testing.assert_allclose(
        np.asarray(corrected.omega_f),
        np.asarray(native.omega_f) * 0.66,
        rtol=1e-10,
    )
