# LISA Astrophysical Backgrounds (LAB)

**Version:** 0.2.0

`lisaastrophysicalbackgrounds` (LAB) is a high-performance Python package designed to calculate the Astrophysical Gravitational Wave Background (AGWB) from by a population of ultra-compact binaries.

Tailored specifically for low-frequency gravitational wave observatories like LISA, LAB is built on top of **JAX** to ensure fast, parallelizable, and JIT-compiled computations. It uses `astropy.cosmology` as the cosmology backend for consistent modeling of the background cosmology. Additionally, it is dependent `lisaconstants` to ensure homogenized usage of constants.

The initial framework of this repository is heavily based on the work of Staelens & Nelemans and Hoffman & Nelemans (see citations). 

**Please note that this project may be subject to significant changes in the upcomming months**

## Key Features

* **High-Performance Computing:** The main physical and cosmological grid computations are vectorized and compiled using `jax.jit` and `jax.vmap` for CPU/GPU acceleration.
* **Component-Separated Background:** Calculates the AGWB by separating it into three distinct contributions:
  1. **Bulk:** Steady continuous emission from evolving binaries.
  2. **Birth:** Emission from systems entering the frequency band.
  3. **Mergers:** Emission from systems merging in the frequency band.
* **Config-Driven:** Fully controlled via a simple YAML configuration file, making parameter space exploration and reproducibility effortless.
* **Flexible Cosmology:** Supports standard and custom cosmologies as implemented in  `astropy`.
* **SFH:** Multiple Star Formation Histories (SFHs) are supported. This included both single- and multi-metallicity dependent SFHs.
* **Variance Estimates:** This package estimates the variance of the GWB strength (`omega_fz`) using an analytical approximation.
* **HDF5 Output:** Saves multi-dimensional redshift-frequency results (`omega_fz`, `N_sources_fz`) natively to HDF5 format.

## Dependencies

These are declared in `pyproject.toml`, which is the authoritative list, and installing the package pulls them in automatically.

* `jax`: all array computation, configured for 64-bit precision on import
* `astropy`: cosmology backend
* `lisaconstants`: homogenized physical constants
* `pandas`: reading population catalogues and SFRD data files
* `h5py`: HDF5 output
* `pyyaml`: configuration files
* `matplotlib`: plotting
* `tqdm`: progress reporting

The package itself contains no `numpy` (JAX is used throughout), though numpy is still installed as a transitive dependency of `jax`, `pandas` and `astropy`.

## Installation

LAB requires **Python ≥ 3.12**. Dependencies are managed with [uv](https://docs.astral.sh/uv/), but a plain `pip` install works just as well.

### CPU

```bash
git clone https://github.com/coen-rondeel/LISA_Astrophysical_Backgrounds.git
cd LISA_Astrophysical_Backgrounds

uv sync            # or: pip install .
```

This installs the CPU build of JAX and is enough to run every example in this repository.

Note that only the package itself is installed: the configuration files, binary catalogues and SFRD data tables live at the repository root and are deliberately **not** bundled into the distribution. Clone the repository (as above) rather than installing from a wheel if you want to run the examples.

### GPU (NVIDIA / CUDA)

The heavy grid computations are JIT-compiled with JAX and benefit substantially from a GPU. Two optional extras are provided, one per CUDA major version:

| Extra | Use when | Installs |
| --- | --- | --- |
| `cuda12x` | your driver reports CUDA 12.x | `jax[cuda12]` |
| `cuda13x` | your driver reports CUDA 13.x | `jax[cuda13]` |

Check which one you need. The version in the top-right of `nvidia-smi` is the highest CUDA release your driver supports:

```bash
nvidia-smi
```

Then install the matching extra:

```bash
uv sync --extra cuda12x            # CUDA 12.x
uv sync --extra cuda13x            # CUDA 13.x
```

or with pip:

```bash
pip install ".[cuda12x]"           # quotes matter in zsh
```

The CUDA wheels bundle their own CUDA runtime, so no system CUDA toolkit is required, only a sufficiently recent NVIDIA driver. They are published for **Linux only** (x86_64 and aarch64); on macOS and Windows, install the CPU build above.

Verify that JAX actually sees the GPU:

```bash
python -c "import jax; print(jax.devices())"
# [CudaDevice(id=0)]   -> GPU in use
# [CpuDevice(id=0)]    -> fell back to CPU
```

If this reports a CPU device on a machine that has a GPU, the usual causes are a CUDA major version that does not match the extra you installed, a mismatch between the `jaxlib` and `jax-cuda*-plugin` versions (JAX prints a warning and silently falls back to CPU), or a node where the CUDA libraries are not loadable. To force CPU deliberately, which is useful for quick tests and for reproducing CI:

```bash
JAX_PLATFORMS=cpu python run.py
```

## Quickstart
The entire calculation pipeline is handled by the `GravitationalWaveBackground` class, which is initialized using a YAML configuration file.

1. **Define your configuration (config.yaml)**
To run the pipeline, you need a configuration file mapping out the cosmology, frequency grid, population catalog, and SFH. Three configurations ship with the repository:

| File | Purpose |
| --- | --- |
| `config.yaml` | Reference template documenting **all** available setting with its allowed values and defaults. Runnable as-is on the bundled single-metallicity mock catalogue. |
| `examples/config_SeBa_singleZ.yaml` | The same single-metallicity example, without the explanatory comments. |
| `examples/config_vanHaaften_multiZ.yaml` | Multi-metallicity example: 6.14M double white dwarfs across 25 metallicities, on a redshift grid uniform in lookback time. This allows us to reproduce the results of Boileau et al. 2025. |

All paths inside a config are resolved **relative to the config file itself**, so a configuration can be run from any working directory.

2. **Run the calculation**
```python
from lisaastrophysicalbackgrounds.gravitational_wave_background import GravitationalWaveBackground

# 1. Initialize the GWB calculator with the path to your config
gwb = GravitationalWaveBackground('config.yaml')

# 2. Calculate Births, Bulk, and Mergers contributions
gwb.calculate_GWB()

# 3. Access results dynamically (if not just relying on the saved HDF5 file)
frequencies = gwb.f_vals
omega_gw = gwb.omega_f
var_omega_gw = gwb.var_f
num_sources = gwb.N_sources_f
```

A worked example, including the variance estimate and the redshift-resolved source counts, is in [`scripts/example_run.ipynb`](scripts/example_run.ipynb).

## Star formation histories
The SFRD ψ(z) is selected with `SFH_name`. Both analytic fits and numerical models are available:

| `SFH_name` | Reference | IMF frame |
| --- | --- | --- |
| `madau_and_dickinson` | Madau & Dickinson (2014) | Salpeter |
| `madau_and_fragos` | Madau & Fragos (2017) | Kroupa |
| `strolger` | Strolger et al. (2004), evaluated in cosmic time | Salpeter |
| `neijssel_2019` | Neijssel et al. (2019) | Kroupa |
| `chruslinska_and_nelemans` | Chruslinska & Nelemans (2019, 2021), numerical | Kroupa |

The numerical `chruslinska_and_nelemans` model reads the tabulated SFRD data from `SFH_path` and selects a variant with `SFRD_model` (`MZ19` moderate, `LZ19`/`HZ19` low/high-Z extreme, `LZ21`/`HZ21` for the inclusion of starburst galaxies). It integrates the numerical 12+log(O/H) distribution over the configured metallicity bins; the conversion scale (`SFH_solar_Z_scale`) and the treatment of star formation outside the metallicity grid (`SFH_coverage`) are configurable. The data extend to z = 10, beyond which ψ = 0.

## Initial mass functions
Each SFRD is calibrated against a particular IMF (last column above), while `total_population_mass` is expressed in the IMF the population synthesis code sampled. The per-binary rate ψ(z, Z) / M_SF(Z) is only meaningful with both in one frame, so LAB converts ψ into the **population's** frame, because the pair (catalogue, M_SF) is self-consistent by construction and is never rescaled.

Declare the population's IMF with `population_IMF` (`kroupa`, `chabrier` or `salpeter`) in the `population` block; the SFH's native frame is looked up from `SFH_name`. Both can be overridden: `SFH_reference_IMF` replaces the lookup, and `SFH_IMF_correction` takes either `"auto"` (the default) or a float used verbatim. Omitting `population_IMF` disables the correction and emits a warning.

Conversion factors come from [Speagle et al. (2014)](https://iopscience.iop.org/article/10.1088/0067-0049/214/2/15) Table C.2 for Chabrier ↔ Kroupa (1.06), and the FUV+IR SFR conversion for Salpeter → Kroupa (0.66); see the [`imf`](src/lisaastrophysicalbackgrounds/imf.py) module for the derivation and the indicator-dependent alternatives. The applied factor and both frames are written to the output HDF5 as attributes.

Note the amplitude this carries: the default pairing of `madau_and_dickinson` (Salpeter) with a Kroupa catalogue applies ×0.66, so results produced before this correction existed are high by a factor 1.5.

## Metallicity
Supplying a list of `SFH_metallicities` (plus a list of `SFH_metallicity_bins`) enables multi-metallicity handling. Star formation is distributed across the grid with the Neijssel et al. (2019) log-normal-in-Z distribution, and each metallicity is normalised by its own star-forming mass, so `total_population_mass` becomes a list with one entry per metallicity, **aligned positionally** to `SFH_metallicities`. The output then additionally carries the metallicity-resolved arrays `omega_fZ`, `omega_fzZ` and their source-count counterparts.

A single `SFH_metallicities` value selects the single-metallicity path, which skips the metallicity distribution entirely and takes a single float for `total_population_mass`.

## Project structure
- `gravitational_wave_background.py`: The main class that performs the GWB calculation. It handles: communication to other submodules, frequency grid initialization, cleaning of the catalogue, computing the birth-bulk-merger GWB components, and exporting/plotting of the data.
- `background_cosmology.py`: Wraps `astropy.cosmology` to initialize the underlying cosmology for the set configuration parameters. 
- `preprocess_catalogue.py`: Handles the importing of the compact binary catalogue. E.g., calculating chirp masses, orbital frequencies, and relavent GW characteristics.
- `star_formation_history.py`: Evaluates start formation rates for a given cosmology explicitly accounting for possible delay times between star formation and binary creation.
- `imf.py`: IMF frames of the SFH models and the conversion factors between them.
- `physics.py`: pure-physics functions.
- `utils.py`: Helper functions for YAML processing and plotting functionalities.

## Contributing
Contributions, issues, and feature requests are welcome!

### Environment: uv
The package is managed with [uv](https://docs.astral.sh/uv/). Work inside the project environment rather than installing dependencies by hand:

```bash
uv sync                          # create/refresh the environment from uv.lock
uv run python run.py             # run anything inside it
uv add <package>                 # add a dependency (updates pyproject.toml and uv.lock)
```

`uv.lock` is committed and must stay in sync with `pyproject.toml`. Continuous integration installs with `uv sync --locked`, which fails on a stale lockfile. If you change dependencies, commit the updated lockfile and check it with:

```bash
uv lock --check
```

### Style: ruff
Linting and formatting are handled by [ruff](https://docs.astral.sh/ruff/), and **the standards are defined in `pyproject.toml`**, so please do not configure ruff locally or add per-file ignores without discussion. The `[tool.ruff]` and `[tool.ruff.lint]` sections currently set:

| Setting | Value | Meaning |
| --- | --- | --- |
| `line-length` | 88 | maximum line length |
| `target-version` | `py310` | syntax modernisation target |
| `select` | `E`, `W` | PEP 8 errors and warnings |
| | `F` | Pyflakes (unused imports, undefined names) |
| | `I` | import sorting |
| | `D` | docstrings, **every public module, class and method needs one** |
| | `UP` | pyupgrade (modern f-strings, builtin generics, ...) |
| `pydocstyle.convention` | `pep257` | docstring layout, with Args/Returns sections |

ruff is deliberately *not* a project dependency, so run it through `uvx`:

```bash
uvx ruff check src tests         # lint; add --fix to apply the safe fixes
uvx ruff format src tests        # format; add --check to only report
```

### Before opening a pull request
```bash
uvx ruff check src tests
uvx ruff format --check src tests
uv run pytest
```
All three must pass. Tests run in CI on both Ubuntu and macOS.

Finally, please ensure that `jax` compatibility is maintained: keep code inside the JIT-compiled kernels branch-free over data (use masks or `jnp.where` rather than Python `if`), and prefer `jnp` over `np` in anything that ends up under `jax.jit` or `jax.vmap`.


## Citation
A citation for this package will be added here in the future, once the accompanying publication is available. In the meantime, please also credit the work this package builds on (Staelens & Nelemans, and Hoffman & Nelemans) and, when using the numerical star formation histories or the multi-metallicity treatment, the corresponding references listed in the sections above. TODO

## License
This project is licensed under the **Apache License, Version 2.0**. The full text is in the [LICENSE](LICENSE.txt) file, and is also available at <http://www.apache.org/licenses/LICENSE-2.0>.
