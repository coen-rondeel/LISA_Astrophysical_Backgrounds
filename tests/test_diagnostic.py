# tests/test_diagnostic.py

import os
from pathlib import Path
from lisaastrophysicalbackgrounds.gravitational_wave_background import GravitationalWaveBackground
from lisaastrophysicalbackgrounds.diagnostic import (
    generate_diagnostic_plots,
    plot_gwb_metallicity_breakdown,
    plot_gwb_redshift_evolution,
    plot_gwb_spectral_components,
    plot_population_properties,
    plot_sfrd_vs_redshift,
)


def test_individual_plotting_functions(mock_project_dir: Path) -> None:
    """Test each plotting function individually using the mock project context."""
    gwb = GravitationalWaveBackground(str(mock_project_dir))
    gwb.calculate_GWB()

    save_dir = Path(gwb.config["global"]["save_directory"])

    # Test plot_population_properties
    pop_path = save_dir / "test_pop.png"
    fig_pop = plot_population_properties(gwb.population, save_path=pop_path)
    assert fig_pop is not None
    assert pop_path.exists()

    # Test plot_gwb_spectral_components
    spec_path = save_dir / "test_spec.png"
    fig_spec = plot_gwb_spectral_components(gwb, save_path=spec_path)
    assert fig_spec is not None
    assert spec_path.exists()

    # Test plot_gwb_redshift_evolution
    redshift_path = save_dir / "test_redshift.png"
    fig_red = plot_gwb_redshift_evolution(gwb, save_path=redshift_path)
    assert fig_red is not None
    assert redshift_path.exists()

    # Test plot_gwb_metallicity_breakdown (mock is single metallicity, should return None)
    met_path = save_dir / "test_met.png"
    fig_met = plot_gwb_metallicity_breakdown(gwb, save_path=met_path)
    assert fig_met is None
    assert not met_path.exists()

    # Test plot_sfrd_vs_redshift
    sfrd_path = save_dir / "test_sfrd.png"
    fig_sfrd = plot_sfrd_vs_redshift(gwb, save_path=sfrd_path)
    assert fig_sfrd is not None
    assert sfrd_path.exists()


def test_generate_diagnostic_plots(mock_project_dir: Path) -> None:
    """Test high-level orchestrator generates the expected diagnostic files."""
    gwb = GravitationalWaveBackground(str(mock_project_dir))
    gwb.calculate_GWB()

    save_dir = Path(gwb.config["global"]["save_directory"])
    plot_paths = generate_diagnostic_plots(gwb, save_directory=save_dir)

    assert "population" in plot_paths
    assert "spectral_components" in plot_paths
    assert "redshift_evolution" in plot_paths
    assert "sfrd_vs_redshift" in plot_paths
    assert "metallicity_breakdown" not in plot_paths  # Single metallicity mock

    assert Path(plot_paths["population"]).exists()
    assert Path(plot_paths["spectral_components"]).exists()
    assert Path(plot_paths["redshift_evolution"]).exists()
    assert Path(plot_paths["sfrd_vs_redshift"]).exists()


def test_pipeline_integration_runs_diagnostics(mock_project_dir: Path) -> None:
    """Test that setting save_diagnostics to True triggers plot generation."""
    # 1. Load config and explicitly enable save_diagnostics
    import yaml
    with open(mock_project_dir) as f:
        config = yaml.safe_load(f)
    config["global"]["save_diagnostics"] = True
    with open(mock_project_dir, "w") as f:
        yaml.safe_dump(config, f)

    gwb = GravitationalWaveBackground(str(mock_project_dir))
    gwb.calculate_GWB()

    # Verify files created in save_directory
    save_dir = Path(gwb.config["global"]["save_directory"])
    pop_name = gwb.config["population"]["population_name"]
    sfh_name = gwb.config["SFH"]["SFH_name"]

    expected_pop = save_dir / f"diagnostic_population_{pop_name}.png"
    expected_spec = save_dir / f"diagnostic_GWB_spectrum_{pop_name}_with_{sfh_name}.png"
    expected_redshift = save_dir / f"diagnostic_GWB_redshift_{pop_name}.png"
    expected_sfrd = save_dir / f"diagnostic_SFRD_{sfh_name}.png"

    assert expected_pop.exists()
    assert expected_spec.exists()
    assert expected_redshift.exists()
    assert expected_sfrd.exists()
