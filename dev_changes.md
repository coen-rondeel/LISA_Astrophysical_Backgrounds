# LISA Astrophysical Backgrounds (LAB) - Dev Branch Modifications

This document summarizes the changes made to the `dev` branch of the `LISA_Astrophysical_Backgrounds` repository to implement the diagnostic plotting subpackage (`lisaastrophysicalbackgrounds.diagnostic`) and fully support JAX-compatible Chruslinska & Nelemans (2019) Star Formation History models.

---

## 1. Directory Structure of Changes

Below are the new and modified files in the codebase:
```text
LISA_Astrophysical_Backgrounds-dev/
├── base_config.yaml (Modified)
├── data/
│   └── SFRDs/
│       ├── LZ19_SFRD_allbins.txt.gz (New)
│       ├── MZ19_SFRD_allbins.txt.gz (New)
│       ├── HZ19_SFRD_allbins.txt.gz (New)
│       ├── LZ21_SFRD_allbins.txt.gz (New)
│       └── HZ21_SFRD_allbins.txt.gz (New)
├── src/
│   └── lisaastrophysicalbackgrounds/
│       ├── gravitational_wave_background.py (Modified)
│       ├── star_formation_history.py (Modified)
│       └── diagnostic/ (New Package)
│           ├── __init__.py (New)
│           └── plotting.py (New)
└── tests/
    └── test_diagnostic.py (New)
```

---

## 2. Modified Existing Files

### A. `src/lisaastrophysicalbackgrounds/gravitational_wave_background.py`
Integrated the diagnostic trigger inside the `save_results` method (lines 720-721). When `save_diagnostics` is enabled, the pipeline automatically generates all relevant plots.

```python
        # Existing GWB plot trigger
        get_GWB_plot(self.f_vals, self.omega_f, self.var_f, save_path=plot_path)

        # New: Diagnostic plotting trigger
        if self.config["global"].get("save_diagnostics", False):
            from .diagnostic import generate_diagnostic_plots
            generate_diagnostic_plots(self)
```

### B. `src/lisaastrophysicalbackgrounds/star_formation_history.py`
Implemented Chruslinska & Nelemans tabulated Star Formation Histories (`chruslinska_and_nelemans`):
- **Data Pre-loading**: Loaded pre-binned $Z$-fractionated `.txt.gz` files containing 6 metallicity columns ($Z = 0.03, 0.02, 0.01, 0.005, 0.001, 0.0001$) and sorted by redshift in ascending order.
- **JAX-compatible Interpolation**: Created JIT-compilable interpolation logic with vectorization support using `jax.vmap` for robust shape handling (broadcasting scalars and processing 1D arrays).

### C. `base_config.yaml`
Added a configuration parameter under `global` to control diagnostic plot generation automatically:

```yaml
global:

      save_results: True

      save_diagnostics: True  # Added to enable/disable diagnostic generation

      save_directory: "./results/"
```

---

## 3. New Files and Subpackages Added

### A. `src/lisaastrophysicalbackgrounds/diagnostic/__init__.py`
Initializes the diagnostic plotting subpackage and exposes the core functions:

```python
"""Diagnostic plotting and pipeline visualization tools."""

from .plotting import (
    generate_diagnostic_plots,
    plot_gwb_metallicity_breakdown,
    plot_gwb_redshift_evolution,
    plot_gwb_spectral_components,
    plot_population_properties,
)

__all__ = [
    "plot_population_properties",
    "plot_gwb_spectral_components",
    "plot_gwb_redshift_evolution",
    "plot_gwb_metallicity_breakdown",
    "generate_diagnostic_plots",
]
```

### B. `src/lisaastrophysicalbackgrounds/diagnostic/plotting.py`
Implements publication-quality plotting functions using modern, premium styling configurations (appropriate colormaps, grid styles, labels, and legends). Key functions:
- `plot_population_properties(population, save_path=None)`: Plots chirp mass distribution, frequency distribution, and mass scatter plot ($m_1$ vs $m_2$).
- `plot_gwb_spectral_components(gwb, save_path=None)`: Plots the total GWB spectrum $\Omega_{\rm GW}(f)$ and breaks it down into **Bulk**, **Birth**, and **Mergers** components.
- `plot_gwb_redshift_evolution(gwb, save_path=None)`: Shows 2D intensity of $\Omega_{\rm GW}(f, z)$ and the 1D integrated GWB strength vs redshift.
- `plot_sfrd_vs_redshift(gwb, save_path=None)`: Features a new **2-panel layout**:
  *   **Left Panel**: Metallicity breakdown of the active Star Formation Rate Density model.
  *   **Right Panel**: Comparison of the Total SFRD across all 5 Chruslinska empirical models (LZ19, MZ19, HZ19, LZ21, HZ21) and the Madau & Dickinson (2014) model.
- `generate_diagnostic_plots(gwb, save_directory=None)`: Orchestrates all diagnostic plots. Crucially, calling `plt.close(fig)` ensures figures are closed after saving to disk, preventing duplicate outputs or leakage in Jupyter Notebooks.

### C. `tests/test_diagnostic.py`
Automated tests to verify correct plot generation and pipeline triggers:
- `test_individual_plotting_functions`: Validates each plotting function individually.
- `test_generate_diagnostic_plots`: Checks high-level orchestration behavior.
- `test_pipeline_integration_runs_diagnostics`: Simulates a pipeline run, verifying automatic diagnostics.

---

## 4. Verification Results

All 8 tests (5 original physics/pipeline tests + 3 new diagnostic tests) pass successfully in the `WD_GWB` environment:
```text
tests/test_diagnostic.py ...                                             [ 37%]
tests/test_physics.py ...                                                [ 75%]
tests/test_pipeline.py ..                                                [100%]
============================== 8 passed in 19.66s ==============================
```

### Generated Files in `results/`
Running the simulation generates:
1. `diagnostic_population_mock_catalogue.png` (Population chirp mass and initial frequencies)
2. `diagnostic_GWB_spectrum_mock_catalogue_with_madau_and_dickinson.png` (Total, Bulk, Birth, and Merger GWB components)
3. `diagnostic_GWB_redshift_mock_catalogue.png` (Redshift-frequency heatmap and peak contribution redshift)
4. `diagnostic_SFRD_*.png` (2-Panel active model SFRD + all-model comparison)
