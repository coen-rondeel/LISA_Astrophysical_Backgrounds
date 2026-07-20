"""Gravitational wave background calculation module."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
from tqdm.auto import tqdm

from .background_cosmology import BackgroundCosmology
from .physics import orbital_freq_from_time, tau_GW
from .preprocess_population import PreprocessPopulation
from .star_formation_history import StarFormationHistory
from .utils import _where, get_config, get_GWB_plot

jax.config.update("jax_enable_x64", True)


class GravitationalWaveBackground:
    """The Gravitational Wave Background Class."""

    def __init__(self, config_file: str) -> None:
        """Initialize the class to calculate the GWB from a binary population.

        Args:
            config_file (str): Full path to the configuration file. Functionality
                assumes other paths are relative to the config file if not absolute.
        """
        # * Making sure that the file paths can be resolved correctly
        config_path = Path(config_file).resolve()
        config_dir = config_path.parent

        self.config: dict = get_config(str(config_file))

        pop_path = Path(self.config["population"]["population_path"])
        if not pop_path.is_absolute():
            self.config["population"]["population_path"] = str(config_dir / pop_path)

        save_dir = Path(self.config["global"]["save_directory"])
        if not save_dir.is_absolute():
            self.config["global"]["save_directory"] = str(config_dir / save_dir)

        Path(self.config["global"]["save_directory"]).mkdir(parents=True, exist_ok=True)

        # * Actual Initialization of the code
        self.get_frequencies()
        self.cosmology = BackgroundCosmology(self.config)
        self.SFH = StarFormationHistory(self.config, self.cosmology.cosmo)

        self.population = PreprocessPopulation(self.config)
        self.clean_population()
        self._snap_population_metallicities()

    def get_frequencies(self) -> None:
        """Set up the frequency relevant quantities according to the configuration."""
        self.f_min = float(self.config["global"]["frequency"]["f_min"])
        self.f_max = float(self.config["global"]["frequency"]["f_max"])
        self.N_fbins = int(float(self.config["global"]["frequency"]["N_fbins"]))

        if self.config["global"]["frequency"]["f_scale"] == "linear":
            f_grid = jnp.linspace(self.f_min, self.f_max, 2 * self.N_fbins + 1)
        elif self.config["global"]["frequency"]["f_scale"] == "log10":
            f_grid = jnp.logspace(
                jnp.log10(self.f_min),
                jnp.log10(self.f_max),
                2 * self.N_fbins + 1,
                base=10,
            )
        else:
            raise NotImplementedError(
                "The f_scale in the config file is currently not (yet) supported. "
                "Supported f_scales are 'linear', 'log10'."
            )

        self.f_grid: jax.Array = f_grid
        self.f_vals: jax.Array = self.f_grid[1::2]
        self.f_bins: jax.Array = self.f_grid[::2]

        self.f_factors: jax.Array = (
            self.f_vals
            * (self.f_bins[1:] ** (2 / 3) - self.f_bins[:-1] ** (2 / 3))
            / (self.f_bins[1:] - self.f_bins[:-1])
        )

    def clean_population(self) -> None:
        """Remove binaries that never reach the minimum frequency within max age."""
        lookbacktime_func = getattr(self.cosmology.cosmo, "lookback_time")
        max_age: jax.Array = (
            jnp.array(lookbacktime_func(self.cosmology.z_max).value) * 1000
        )  # in Myr
        tau_max: jax.Array = tau_GW(
            2 * self.population.nu0, self.f_min, self.population.K_factor
        )

        # ? + self.population.t0 not certain if this should be included
        mask: jax.Array = tau_max < max_age
        # TODO: add metallicity filter for binaries not in targeted metallicities

        self.population.t0 = self.population.t0[mask]
        self.population.M_ch = self.population.M_ch[mask]
        self.population.M_ch_pow = self.population.M_ch_pow[mask]
        self.population.K_factor = self.population.K_factor[mask]
        self.population.nu0 = self.population.nu0[mask]
        self.population.numax = self.population.numax[mask]
        self.population.merger_time = self.population.merger_time[mask]

        if self.population.Z is not None:
            self.population.Z = self.population.Z[mask]

    def _snap_population_metallicities(self) -> None:
        """Map each binary onto the configured metallicity grid.

        The GWB kernels group binaries by exact match to ``SFH_metallicities``.
        Binning each population metallicity onto the configured grid (via
        ``SFH_metallicity_bins``) keeps the grid basis config-owned and prevents
        floating-point mismatches from silently dropping sources.
        """
        if self.population.Z is None:
            return

        centers: jax.Array = jnp.sort(
            jnp.atleast_1d(jnp.array(self.config["SFH"]["SFH_metallicities"]))
        )
        bins = self.config["SFH"].get("SFH_metallicity_bins")
        if bins is None or centers.size == 1:
            self.population.Z = jnp.full_like(self.population.Z, centers[0])
            return

        edges: jax.Array = jnp.sort(jnp.atleast_1d(jnp.array(bins)))
        idx: jax.Array = jnp.clip(
            jnp.searchsorted(edges, self.population.Z, side="right") - 1,
            0,
            centers.size - 1,
        )
        self.population.Z = centers[idx]

    def _get_population_batches(
        self, batch_size: int
    ) -> Iterator[dict[str, jax.Array]]:
        """Yield uniform-sized, padded batches to prevent JIT recompilation."""
        total_size: int = len(self.population.nu0)
        minimal_batch_size: int = min(batch_size, total_size)

        for start_idx in range(0, total_size, minimal_batch_size):
            end_idx = min(start_idx + minimal_batch_size, total_size)
            pad_len = minimal_batch_size - (end_idx - start_idx)

            def pad_array(arr: jax.Array | None) -> jax.Array:
                if arr is None:
                    return jnp.zeros(minimal_batch_size)  # Dummy array for JAX
                chunk = arr[start_idx:end_idx]
                return (
                    jnp.pad(chunk, (0, pad_len), constant_values=0)
                    if pad_len > 0
                    else chunk
                )

            valid_mask: jax.Array = jnp.arange(minimal_batch_size) < (
                end_idx - start_idx
            )

            yield {
                "nu0": pad_array(self.population.nu0),
                "numax": pad_array(self.population.numax),
                "t0": pad_array(self.population.t0),
                "K_factor": pad_array(self.population.K_factor),
                "M_ch_pow": pad_array(self.population.M_ch_pow),
                "merger_time": pad_array(self.population.merger_time),
                "Z": pad_array(self.population.Z),
                "valid_mask": valid_mask,
            }

    # * ===================== Main calculation of the GWB =====================
    def calculate_GWB(self) -> None:
        """Calculate the gravitational wave background from the population."""
        has_Z = self.population.Z is not None
        if has_Z:
            assert self.population.Z is not None
            self.unique_Zs = jnp.array(self.config["SFH"]["SFH_metallicities"])
        else:
            self.unique_Zs = jnp.array([0.0])

        # * pre-compiling kernels
        bulk_kernel = self._build_bulk_kernel(self.unique_Zs, has_Z)
        birth_kernel = self._build_birth_kernel(self.unique_Zs, has_Z)
        merger_kernel = self._build_merger_kernel(self.unique_Zs, has_Z)

        # * initialize zero-accumulators
        output_shape = (self.N_fbins, self.cosmology.N_zbins, len(self.unique_Zs))
        bulk_raw = birth_raw = merger_raw = (
            jnp.zeros(output_shape),
            jnp.zeros(output_shape),
            jnp.zeros(output_shape),
        )

        batch_size = int(self.config["global"].get("batch_size", 1_000_000))
        tot_size = len(self.population.nu0)
        n_batches = (
            (tot_size + min(batch_size, tot_size) - 1) // min(batch_size, tot_size)
            if tot_size > 0
            else 0
        )

        # * loop over batches making use of jax.jit compilation
        for batch in tqdm(
            self._get_population_batches(batch_size),
            total=n_batches,
            desc="GWB",
            unit="batch",
        ):
            bulk_raw_batch = bulk_kernel(batch)
            bulk_raw = tuple(r + bu for r, bu in zip(bulk_raw, bulk_raw_batch))

            birth_raw_batch = birth_kernel(batch)
            birth_raw = tuple(r + bi for r, bi in zip(birth_raw, birth_raw_batch))

            merger_raw_batch = merger_kernel(batch)
            merger_raw = tuple(r + me for r, me in zip(merger_raw, merger_raw_batch))

        tot_pop_mass = self.population.total_population_mass
        tot_mass_broadcast = (
            tot_pop_mass[None, None, :]
            if isinstance(tot_pop_mass, jax.Array)
            else tot_pop_mass
        )

        assert len(bulk_raw) == 3 and len(birth_raw) == 3 and len(merger_raw) == 3
        self._apply_cosmological_weighting(
            bulk_raw, birth_raw, merger_raw, tot_mass_broadcast
        )
        self.combine_contributions()

        if self.config["global"].get("save_results", False):
            self.save_results()

    def _build_bulk_kernel(
        self, unique_Zs: jax.Array, has_Z: bool
    ) -> Callable[[dict[str, jax.Array]], tuple[jax.Array, jax.Array, jax.Array]]:
        """Calculate the total bulk contribution to the GWB."""

        @jax.jit
        def compute_bulk(
            batch_now: dict[str, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:

            def bulk_z_f(
                z: jax.Array,
                age: jax.Array,
                t_max_z: jax.Array,
                f_low_obs: jax.Array,
                f_high_obs: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                """Calculate the partial bulk contribution to a redshift and freq bin.

                Additionally, computes across metallicity bins if N_Zbins > 1.

                Args:
                    z (jax.Array): Redshift value of the considered bin.
                    age (jax.Array): Age of universe at the considered bin in Myr.
                    t_max_z (jax.Array): Relative age of the universe in Myr.
                    f_low_obs (jax.Array): Lower frequency of the bin in Hz.
                    f_high_obs (jax.Array): Upper frequency of the bin in Hz.

                Returns:
                    Tuple[jax.Array, jax.Array, jax.Array]: Partial bulk contribution
                    for Omega, number of systems, and variance in the GWB strength.
                """
                f_low_zbin: jax.Array = f_low_obs * (1 + z)
                f_high_zbin: jax.Array = f_high_obs * (1 + z)

                # mask to focus on systems that contribute to this bin
                mask: jax.Array = (
                    (2 * batch_now["nu0"] <= f_low_zbin)
                    & (2 * batch_now["numax"] >= f_high_zbin)
                    & batch_now["valid_mask"]
                )
                time_since_ZAMS: jax.Array = (
                    tau_GW(2 * batch_now["nu0"], f_high_zbin, batch_now["K_factor"])
                    + batch_now["t0"]
                )

                # mask to only include systems not older than universe at max_z
                mask: jax.Array = mask & (time_since_ZAMS <= t_max_z)

                psi: jax.Array = self.SFH.delayed_SFH(
                    age, time_since_ZAMS, batch_now["Z"] if has_Z else None
                )
                tau_bin: jax.Array = tau_GW(
                    f_low_zbin, f_high_zbin, batch_now["K_factor"]
                )

                # contribution to the GWB of each binary
                num_syst_weights: jax.Array = _where(mask, psi * tau_bin * 1e6, 0.0)
                omega_weights: jax.Array = _where(
                    mask, psi * batch_now["M_ch_pow"], 0.0
                )
                num_syst_safe: jax.Array = _where(
                    num_syst_weights > 0, num_syst_weights, 1.0
                )
                var_weights = _where(
                    num_syst_weights > 0,
                    omega_weights * omega_weights / num_syst_safe,
                    0.0,
                )

                # summing all binaries according to their metallicity
                def sum_by_Z(Z_val):
                    Z_mask = (batch_now["Z"] == Z_val) if has_Z else jnp.ones_like(mask)
                    return (
                        jnp.sum(_where(Z_mask, omega_weights, 0.0)),
                        jnp.sum(_where(Z_mask, num_syst_weights, 0.0)),
                        jnp.sum(_where(Z_mask, var_weights, 0.0)),
                    )

                return jax.vmap(sum_by_Z)(unique_Zs)

            vmap_z = jax.vmap(bulk_z_f, in_axes=(0, 0, 0, None, None))

            def scan_f(carry, f_edges):
                return carry, vmap_z(
                    self.cosmology.z_vals,
                    self.cosmology.ages,
                    self.cosmology.z_time_since_z_max,
                    f_edges[0],
                    f_edges[1],
                )

            _, out = jax.lax.scan(scan_f, None, (self.f_bins[:-1], self.f_bins[1:]))
            return out

        return compute_bulk

    def _build_birth_kernel(
        self, unique_Zs: jax.Array, has_Z: bool
    ) -> Callable[[dict[str, jax.Array]], tuple[jax.Array, jax.Array, jax.Array]]:
        """Calculate the total birth contribution to the GWB."""

        @jax.jit
        def compute_births(
            batch_now: dict[str, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:

            def birth_z(
                z: jax.Array, age: jax.Array, t_max_z: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                """Calculate the partial birth contribution to a redshift and freq bin.

                Args:
                    z (jax.Array): Redshift value of the considered bin.
                    age (jax.Array): Age of universe at the considered bin in Myr.
                    t_max_z (jax.Array): Relative age of the universe in Myr.

                Returns:
                    Tuple[jax.Array, jax.Array, jax.Array]: Partial birth contribution
                    for Omega, number of systems, and variance in the GWB strength.
                """
                f_birth_obs: jax.Array = (2 * batch_now["nu0"]) / (1 + z)
                psi: jax.Array = self.SFH.delayed_SFH(
                    age, batch_now["t0"], batch_now["Z"] if has_Z else None
                )
                bin_idx = jnp.searchsorted(self.f_bins, f_birth_obs, side="right") - 1

                # mask for systems born within frequency range & valid age
                within_band: jax.Array = (bin_idx >= 0) & (bin_idx < self.N_fbins)
                age_OK: jax.Array = batch_now["t0"] <= t_max_z
                mask: jax.Array = within_band & age_OK & batch_now["valid_mask"]

                freq_map = _where(mask, bin_idx, 0)
                f_low_obs, f_high_obs = self.f_bins[freq_map], self.f_bins[freq_map + 1]

                max_evolve_time = t_max_z - batch_now["t0"]
                tau_to_upper_edge = tau_GW(
                    2 * batch_now["nu0"], f_high_obs * (1 + z), batch_now["K_factor"]
                )

                # mask for systems that can reach upper edge within available time
                mask_reach_edge: jax.Array = tau_to_upper_edge >= max_evolve_time
                tau_in_bin = _where(mask_reach_edge, max_evolve_time, tau_to_upper_edge)

                # calculate the reached frequency after evolving for tau_in_bin
                nu_reached_zbin = _where(
                    mask_reach_edge,
                    orbital_freq_from_time(
                        batch_now["nu0"], max_evolve_time, batch_now["K_factor"]
                    ),
                    f_high_obs * (1 + z) * 0.5,
                )
                nu_reached_zbin = (
                    nu_reached_zbin[0]
                    if isinstance(nu_reached_zbin, tuple)
                    else nu_reached_zbin
                )

                # calculate birth frequency factors
                freq_fac = (
                    jnp.square(jnp.cbrt(nu_reached_zbin))
                    - jnp.square(jnp.cbrt(batch_now["nu0"]))
                ) / (f_high_obs - f_low_obs)

                # calculate omega and N_sources contributions of each system
                omega_weights: jax.Array = _where(
                    mask,
                    self.f_vals[freq_map]
                    * batch_now["M_ch_pow"]
                    * freq_fac
                    * (1 / (1 + z))
                    * psi,
                    0.0,
                )
                num_syst_weights: jax.Array = _where(mask, psi * tau_in_bin * 1e6, 0.0)
                num_syst_safe: jax.Array = _where(
                    num_syst_weights > 0, num_syst_weights, 1.0
                )
                var_weights = _where(
                    num_syst_weights > 0,
                    omega_weights * omega_weights / num_syst_safe,
                    0.0,
                )

                # map to frequency bin and sum systems with the same metallicity
                def sum_by_Z(Z_val):
                    Z_mask = (batch_now["Z"] == Z_val) if has_Z else jnp.ones_like(mask)
                    return (
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, omega_weights, 0.0)),
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, num_syst_weights, 0.0)),
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, var_weights, 0.0)),
                    )

                partial_o, partial_n, partial_v = jax.vmap(sum_by_Z)(unique_Zs)
                return partial_o.T, partial_n.T, partial_v.T

            out = jax.vmap(birth_z, in_axes=(0, 0, 0))(
                self.cosmology.z_vals,
                self.cosmology.ages,
                self.cosmology.z_time_since_z_max,
            )
            return (
                jnp.transpose(out[0], (1, 0, 2)),
                jnp.transpose(out[1], (1, 0, 2)),
                jnp.transpose(out[2], (1, 0, 2)),
            )

        return compute_births

    def _build_merger_kernel(
        self, unique_Zs: jax.Array, has_Z: bool
    ) -> Callable[[dict[str, jax.Array]], tuple[jax.Array, jax.Array, jax.Array]]:
        """Calculate the total merger contribution to the GWB."""

        @jax.jit
        def compute_mergers(
            batch_now: dict[str, jax.Array],
        ) -> tuple[jax.Array, jax.Array, jax.Array]:

            def merger_z(
                z: jax.Array, age: jax.Array, t_max_z: jax.Array
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                """Calculate the partial merger contribution to a redshift and freq bin.

                Args:
                    z (jax.Array): Redshift value of the considered bin.
                    age (jax.Array): Age of universe at the considered bin in Myr.
                    t_max_z (jax.Array): Relative age of the universe in Myr.

                Returns:
                    Tuple[jax.Array, jax.Array, jax.Array]: Partial merger contribution
                    for Omega, number of systems, and variance in the GWB strength.
                """
                evolve_time: jax.Array = t_max_z - batch_now["t0"]
                merger_reached: jax.Array = batch_now["merger_time"] <= evolve_time
                f_now_zbin = _where(
                    merger_reached,
                    2 * batch_now["numax"],
                    2
                    * orbital_freq_from_time(
                        batch_now["nu0"], evolve_time, batch_now["K_factor"]
                    ),
                )
                f_now_zbin = (
                    f_now_zbin[0] if isinstance(f_now_zbin, tuple) else f_now_zbin
                )

                f_now_obs: jax.Array = f_now_zbin / (1 + z)
                bin_idx = jnp.searchsorted(self.f_bins, f_now_obs, side="right") - 1

                # mask for systems that merge within frequency range & binary is born
                within_band: jax.Array = (bin_idx >= 0) & (bin_idx < self.N_fbins)
                binary_born: jax.Array = evolve_time >= 0
                mask: jax.Array = within_band & binary_born & batch_now["valid_mask"]

                # do not consider binaries that merge outside of the frequency range
                mask: jax.Array = mask & (f_now_obs <= 2 * batch_now["numax"] / (1 + z))

                # do not overcount mergers included in the birth contribution
                f_low_obs: jax.Array = self.f_bins[_where(mask, bin_idx, 0)]
                f_low_zbin: jax.Array = f_low_obs * (1 + z)
                mask: jax.Array = mask & (2 * batch_now["nu0"] < f_low_zbin)

                # physics for the binaries that reached the merger
                Z_arg = batch_now["Z"] if has_Z else None
                psi_merged = self.SFH.delayed_SFH(age, batch_now["merger_time"], Z_arg)

                # same for the non-merged binaries
                psi_non_merged = self.SFH.delayed_SFH(
                    age,
                    tau_GW(2 * batch_now["nu0"], f_low_zbin, batch_now["K_factor"]),
                    Z_arg,
                )
                psi = _where(merger_reached, psi_merged, psi_non_merged)

                freq_map = _where(mask, bin_idx, 0)
                f_low_obs, f_high_obs = self.f_bins[freq_map], self.f_bins[freq_map + 1]

                # frequency factor for merger case:
                nu_upp_23 = _where(
                    merger_reached,
                    jnp.square(jnp.cbrt(batch_now["numax"])),
                    jnp.square(jnp.cbrt(f_now_zbin * 0.5)),
                )
                freq_fac = (nu_upp_23 - jnp.square(jnp.cbrt(f_low_zbin * 0.5))) / (
                    f_high_obs - f_low_obs
                )

                tau_in_bin = _where(
                    merger_reached,
                    tau_GW(f_low_zbin, 2 * batch_now["numax"], batch_now["K_factor"]),
                    evolve_time
                    - tau_GW(2 * batch_now["nu0"], f_low_zbin, batch_now["K_factor"]),
                )
                tau_in_bin = (
                    tau_in_bin[0] if isinstance(tau_in_bin, tuple) else tau_in_bin
                )

                # calculate omega and N_sources contributions of each system
                omega_weights: jax.Array = _where(
                    mask,
                    self.f_vals[freq_map]
                    * batch_now["M_ch_pow"]
                    * freq_fac
                    * (1 / (1 + z))
                    * psi,
                    0.0,
                )
                num_syst_weights: jax.Array = _where(mask, psi * tau_in_bin * 1e6, 0.0)
                num_syst_safe: jax.Array = _where(
                    num_syst_weights > 0, num_syst_weights, 1.0
                )
                var_weights = _where(
                    num_syst_weights > 0,
                    omega_weights * omega_weights / num_syst_safe,
                    0.0,
                )

                # map to frequency bin and sum systems with the same metallicity
                def sum_by_Z(Z_val):
                    Z_mask = (batch_now["Z"] == Z_val) if has_Z else jnp.ones_like(mask)
                    return (
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, omega_weights, 0.0)),
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, num_syst_weights, 0.0)),
                        jnp.zeros(self.N_fbins)
                        .at[freq_map]
                        .add(_where(Z_mask, var_weights, 0.0)),
                    )

                partial_o, partial_n, partial_v = jax.vmap(sum_by_Z)(unique_Zs)
                return partial_o.T, partial_n.T, partial_v.T

            out = jax.vmap(merger_z, in_axes=(0, 0, 0))(
                self.cosmology.z_vals,
                self.cosmology.ages,
                self.cosmology.z_time_since_z_max,
            )
            return (
                jnp.transpose(out[0], (1, 0, 2)),
                jnp.transpose(out[1], (1, 0, 2)),
                jnp.transpose(out[2], (1, 0, 2)),
            )

        return compute_mergers

    def _apply_cosmological_weighting(
        self,
        raw_bulk: tuple[jax.Array, jax.Array, jax.Array],
        raw_birth: tuple[jax.Array, jax.Array, jax.Array],
        raw_merger: tuple[jax.Array, jax.Array, jax.Array],
        tot_mass_broadcast: float | jax.Array,
    ) -> None:
        """Apply precalculated cosmological scaling to raw accumulators."""
        pre_bulk = 8.10e-9 / tot_mass_broadcast
        pre_bm = 1.28e-8 / tot_mass_broadcast

        c_omega_bulk = (
            pre_bulk
            * ((1 + self.cosmology.z_vals) ** (-4 / 3))[None, :, None]
            * self.cosmology.z_widths[None, :, None]
            * self.f_factors[:, None, None]
        )
        c_num_syst = (
            (1.0 / tot_mass_broadcast)
            * 4
            * jnp.pi
            * (self.cosmology.DC_vals**2)[None, :, None]
            * self.cosmology.z_widths[None, :, None]
        )
        safe_c_num_syst = _where(c_num_syst > 0, c_num_syst, 1.0)
        c_var_bulk = _where(c_num_syst > 0, c_omega_bulk**2 / safe_c_num_syst, 0.0)

        # pre-factors of birth and merger are identical
        c_omega_bm = (
            pre_bm
            * ((1 + self.cosmology.z_vals) ** (-1))[None, :, None]
            * self.cosmology.z_widths[None, :, None]
        )
        c_var_bm = _where(c_num_syst > 0, c_omega_bm**2 / safe_c_num_syst, 0.0)

        self.omega_bulk_fzZ, self.N_sources_bulk_fzZ, self.var_bulk_fzZ = (
            c_omega_bulk * raw_bulk[0],
            c_num_syst * raw_bulk[1],
            c_var_bulk * raw_bulk[2],
        )
        self.omega_birth_fzZ, self.N_sources_birth_fzZ, self.var_birth_fzZ = (
            c_omega_bm * raw_birth[0],
            c_num_syst * raw_birth[1],
            c_var_bm * raw_birth[2],
        )
        self.omega_merger_fzZ, self.N_sources_merger_fzZ, self.var_merger_fzZ = (
            c_omega_bm * raw_merger[0],
            c_num_syst * raw_merger[1],
            c_var_bm * raw_merger[2],
        )

    def combine_contributions(self) -> None:
        """Combine the different contributions into one GWB."""
        self.omega_fzZ: jax.Array = (
            self.omega_bulk_fzZ + self.omega_birth_fzZ + self.omega_merger_fzZ
        )
        self.N_sources_fzZ: jax.Array = (
            self.N_sources_bulk_fzZ
            + self.N_sources_birth_fzZ
            + self.N_sources_merger_fzZ
        )
        self.var_fzZ: jax.Array = (
            self.var_bulk_fzZ + self.var_birth_fzZ + self.var_merger_fzZ
        )

        self.omega_fZ = jnp.sum(self.omega_fzZ, axis=1)
        self.N_sources_fZ = jnp.sum(self.N_sources_fzZ, axis=1)
        self.var_fZ = jnp.sum(self.var_fzZ, axis=1)

        self.omega_fz = jnp.sum(self.omega_fzZ, axis=2)
        self.N_sources_fz = jnp.sum(self.N_sources_fzZ, axis=2)
        self.var_fz = jnp.sum(self.var_fzZ, axis=2)

        self.omega_f = jnp.sum(self.omega_fz, axis=1)
        self.N_sources_f = jnp.sum(self.N_sources_fz, axis=1)
        self.var_f = jnp.sum(self.var_fz, axis=1)

    def save_results(self) -> None:
        """Save the results to an HDF5 file and generate a plot."""
        print("Saving results at " + self.config["global"]["save_directory"])
        save_directory: Path = Path(self.config["global"]["save_directory"])
        pop_name = self.config["population"]["population_name"]
        data_filename: str = f"{pop_name}_gwb_results.h5"
        output_path: Path = save_directory / data_filename

        with h5py.File(output_path, "w") as hf:
            frequency_data = hf.create_dataset("f_vals", data=self.f_vals)
            frequency_data.attrs["f_min"] = self.f_min
            frequency_data.attrs["f_max"] = self.f_max
            frequency_data.attrs["N_fbins"] = self.N_fbins
            frequency_data.attrs["frequency_scale"] = self.config["global"][
                "frequency"
            ]["f_scale"]

            redshift_data = hf.create_dataset("z_vals", data=self.cosmology.z_vals)
            redshift_data.attrs["z_min"] = self.cosmology.z_min
            redshift_data.attrs["z_max"] = self.cosmology.z_max
            redshift_data.attrs["N_zbins"] = self.cosmology.N_zbins
            redshift_data.attrs["z_scale"] = self.cosmology.z_scale
            redshift_data.attrs["cosmology"] = str(self.cosmology.cosmo)

            hf.attrs["population"] = pop_name
            hf.create_dataset("omega_fz", data=self.omega_fz)
            hf.create_dataset("N_sources_fz", data=self.N_sources_fz)
            hf.create_dataset("var_fz", data=self.var_fz)

            hf.create_dataset("omega_f", data=self.omega_f)
            hf.create_dataset("N_sources_f", data=self.N_sources_f)
            hf.create_dataset("var_f", data=self.var_f)

            if len(self.unique_Zs) > 1:
                hf.attrs["unique_Zs"] = self.unique_Zs
                hf.create_dataset("omega_fzZ", data=self.omega_fzZ)
                hf.create_dataset("N_sources_fzZ", data=self.N_sources_fzZ)
                hf.create_dataset("var_fzZ", data=self.var_fzZ)

                hf.create_dataset("omega_fZ", data=self.omega_fZ)
                hf.create_dataset("N_sources_fZ", data=self.N_sources_fZ)
                hf.create_dataset("var_fZ", data=self.var_fZ)

        sfh_name = self.config["SFH"]["SFH_name"]
        plot_filename: str = f"GWB_for_{pop_name}_with_{sfh_name}.png"
        plot_path: Path = save_directory / plot_filename
        get_GWB_plot(self.f_vals, self.omega_f, self.var_f, save_path=plot_path)

        if self.config["global"].get("save_diagnostics", False):
            from .diagnostic import generate_diagnostic_plots
            generate_diagnostic_plots(self)

