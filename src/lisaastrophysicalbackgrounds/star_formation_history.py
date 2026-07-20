"""Module containing the classes and models for Star Formation History (SFH)."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from astropy.cosmology import Cosmology


class StarFormationHistory:
    """SFH class instance."""

    def __init__(self, config: dict, cosmo: Cosmology) -> None:
        """Initialize the SFH class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
            cosmo (Cosmology): The considered cosmology as initialized in the
                BackgroundCosmolgy class.
        """
        self._config: dict = config
        self.get_metallicities()

        # * Setup high resolution grid for interpolation
        self.z_high_res = jnp.linspace(0, 10, int(1e4))
        age_func = getattr(cosmo, "age")
        self.age_high_res = jnp.array(age_func(self.z_high_res).value * 1000)  # in Myr
        self._flip_age = jnp.flip(self.age_high_res)
        self._flip_z = jnp.flip(self.z_high_res)

        # * Get SFH function
        sfh_name = self._config["SFH"]["SFH_name"]

        if sfh_name == "chruslinska_and_nelemans":
            self._setup_chruslinska_data()

        self._psi_function: Callable = getattr(self, sfh_name)

    def get_metallicities(self) -> None:
        """Set up the metallicity grid using the config file if N_Zs > 1."""
        self.SFH_Zs = jnp.atleast_1d(
            jnp.array(self._config["SFH"]["SFH_metallicities"])
        )

        if len(self.SFH_Zs) > 1:
            self.SFH_Z_bins = jnp.atleast_1d(
                jnp.array(self._config["SFH"]["SFH_metallicity_bins"])
            )
            assert len(self.SFH_Z_bins) == len(self.SFH_Zs) + 1, (
                "Metallicity grid not initiallized correctly, please check."
            )
            self._Z_centers = jnp.sort(self.SFH_Zs)
            self._Z_edges = jnp.sort(self.SFH_Z_bins)
        else:
            self.SFH_Z_bins = None
            self._Z_centers = self.SFH_Zs
            self._Z_edges = None

    def delayed_SFH(
        self, age: jax.Array, delay: jax.Array, metallicity: jax.Array | None = None
    ) -> jax.Array:
        """Calculate the delayed star formation rate for a given universe state.

        Args:
            age (jax.Array): The age of the universe in Myr.
            delay (jax.Array): The delay of the considered star since formation.
                I.e., the difference between the formation time of the stars
                and time of creation of the GW binary in Myr.
            metallicity (jax.Array): The metallicity of the binary.

        Returns:
            jax.Array: The delayed star formation history in Msol / yr / Mpc^3.
        """
        age_at_star_formation: jax.Array = age - delay  # in Myr

        z_at_star_formation = jnp.interp(
            age_at_star_formation, self._flip_age, self._flip_z
        )

        if self._config["SFH"]["SFH_name"] == "strolger":
            return self._psi_function(
                age_at_star_formation, z_at_star_formation, metallicity
            )

        return self._psi_function(z_at_star_formation, metallicity)

    def madau_and_dickinson(
        self, redshifts: jax.Array, metallicity: jax.Array | None
    ) -> jax.Array:
        """Evaluate the Madau and Dickinson SFR.

        Ref: https://doi.org/10.1146/annurev-astro-081811-125615

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Madau and Dickinson SFR in Msol / yr / Mpc^3
        """

        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.015 * jnp.power(1 + z, 2.7) / (1 + jnp.power((1 + z) / 2.9, 5.6))

        sfr_z = SFH_function(redshifts)

        if metallicity is None:
            return sfr_z

        p_Z = self.metallicity_distribution(redshifts, metallicity)

        return sfr_z * p_Z

    def madau_and_fragos(
        self, redshifts: jax.Array, metallicity: jax.Array | None
    ) -> jax.Array:
        """Evaluate the Madau and Fragos SFR.

        Ref: https://doi.org/10.3847/1538-4357/aa6af9

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Madau and Fragos SFR in Msol / yr / Mpc^3
        """

        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.01 * jnp.power(1 + z, 2.6) / (1 + jnp.power((1 + z) / 3.2, 6.2))

        sfr_z = SFH_function(redshifts)

        if metallicity is None:
            return sfr_z

        p_Z = self.metallicity_distribution(redshifts, metallicity)

        return sfr_z * p_Z

    def strolger(
        self, ages: jax.Array, redshifts: jax.Array, metallicity: jax.Array | None
    ) -> jax.Array:
        """Evaluate the Strolger et al. 2004 SFR.

        Ref: https://doi.org/10.1086/422901

        Args:
            ages (jax.Array): The considered ages of the Universe
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Strolger SFR in Msol / yr / Mpc^3
        """

        def SFH_function(t: jax.Array) -> jax.Array:
            return 0.182 * (
                jnp.power(t, 1.26) * jnp.exp(-t / 1.865)
                + 0.071 * jnp.exp(0.071 * (t - 13.47) / 1.865)
            )

        sfr_t = SFH_function(ages * 1e-3)

        if metallicity is None:
            return sfr_t

        p_Z = self.metallicity_distribution(redshifts, metallicity)

        return sfr_t * p_Z

    def metallicity_distribution(
        self, z: jax.Array, metallicity: jax.Array
    ) -> jax.Array:
        """Calculate the fraction of star formation at each binary's metallicity.

        Follows Boileau et al. 2025 (Eqs. 15-18): the Neijssel et al. 2019
        log-normal dP/dZ weighted by the metallicity-bin width and renormalized
        over the metallicity grid, so that all star formation is distributed
        across the simulated metallicities.

        Args:
            z (jax.Array): Redshift at the moment of star formation.
            metallicity (jax.Array): The metallicity of each binary.

        Returns:
            jax.Array: The dimensionless metallicity weight per binary.
        """
        weight: jax.Array = self._neijssel_metallicity_pdf(
            z, metallicity
        ) * self._metallicity_bin_width(metallicity)

        norm: jax.Array = jnp.sum(
            jax.vmap(
                lambda Z_c: (
                    self._neijssel_metallicity_pdf(z, Z_c)
                    * self._metallicity_bin_width(Z_c)
                )
            )(self._Z_centers),
            axis=0,
        )

        return weight / jnp.where(norm > 0, norm, 1.0)

    def _neijssel_metallicity_pdf(
        self, z: jax.Array, metallicity: jax.Array
    ) -> jax.Array:
        """Evaluate the Neijssel et al. 2019 metallicity distribution dP/dZ.

        Ref: https://arxiv.org/abs/1906.08136 (Equations 7 and 8)

        Args:
            z (jax.Array): The considered redshifts.
            metallicity (jax.Array): The metallicity of each binary.

        Returns:
            jax.Array: The log-normal metallicity density dP/dZ in units of 1/Z.
        """
        mean_Z: jax.Array = 0.035 * jnp.power(10.0, -0.23 * z)
        sigma: float = 0.39
        mu: jax.Array = jnp.log(mean_Z) - jnp.square(sigma) / 2.0
        ln_Z: jax.Array = jnp.log(metallicity)

        return (1.0 / (metallicity * sigma * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
            -0.5 * jnp.square((ln_Z - mu) / sigma)
        )

    def _metallicity_bin_width(self, metallicity: jax.Array) -> jax.Array:
        """Return the width in Z of the metallicity bin containing each value."""
        if self._Z_edges is None:
            return jnp.ones_like(metallicity)

        idx: jax.Array = jnp.clip(
            jnp.searchsorted(self._Z_edges, metallicity, side="right") - 1,
            0,
            self._Z_edges.size - 2,
        )
        return self._Z_edges[idx + 1] - self._Z_edges[idx]

    def neijssel_2019(self, z: jax.Array, metallicity: jax.Array | None) -> jax.Array:
        """Calculate the SFRD and Metallicity distribution from Neijssel et al. 2019.

        Ref: https://arxiv.org/abs/1906.08136 (Equations 6, 7, and 8)

        Args:
            z (jax.Array): The considered redshifts.
            metallicity (Optional[jax.Array]): The metallicity of the binary.

        Returns:
            jax.Array: The evaluated SFRD in Msol / yr / Mpc^3 (per unit Z if
                metallicity is provided).
        """
        # (Eq. 6 from Neijssel et al. 2019)
        sfr_z = (
            0.01 * jnp.power(1.0 + z, 2.77) / (1.0 + jnp.power((1.0 + z) / 2.9, 4.7))
        )

        if metallicity is None:
            return sfr_z

        return sfr_z * self.metallicity_distribution(z, metallicity)

    def _setup_chruslinska_data(self) -> None:
        import gzip
        import pandas as pd
        from pathlib import Path

        sfh_path = Path(self._config['SFH']['SFH_path'])
        # Resolve relative to the config file's directory if not absolute
        if not sfh_path.is_absolute():
            config_dir = Path(self._config.get("global", {}).get("_config_dir", "."))
            sfh_path = config_dir / sfh_path

        MAP_DAT_TO_ALLBINS = {
            "moderate_FOH_z_dM.dat": "MZ19_SFRD_allbins.txt.gz",
            "low-Z_extreme_FOH_z_dM.dat.txt": "LZ19_SFRD_allbins.txt.gz",
            "high-Z_extreme_FOH_z_dM.dat.txt": "HZ19_SFRD_allbins.txt.gz",
            "204f14SBBiC_FMR270_FOH_z_dM.dat": "LZ21_SFRD_allbins.txt.gz",
            "302f14SBBiC_FMR270_FOH_z_dM.dat": "HZ21_SFRD_allbins.txt.gz"
        }

        dat_name = sfh_path.name
        allbins_name = MAP_DAT_TO_ALLBINS.get(dat_name, None)
        if allbins_name is None:
            raise ValueError(f"Unknown Chruslinska file: {dat_name}")

        allbins_path = sfh_path.parent / allbins_name

        with gzip.open(str(allbins_path), 'rt') as f:
            df = pd.read_csv(f)

        # Sort by redshift in ascending order for JAX interpolation compatibility
        df_sorted = df.sort_values('redshift').reset_index(drop=True)

        self._chruslinska_z = jnp.array(df_sorted['redshift'].values)
        self._chruslinska_sfrds = jnp.array([
            df_sorted['0'].values,  # Z=0.03 (z03)
            df_sorted['1'].values,  # Z=0.02 (z02)
            df_sorted['2'].values,  # Z=0.01 (z01)
            df_sorted['3'].values,  # Z=0.005 (z005)
            df_sorted['4'].values,  # Z=0.001 (z001)
            df_sorted['5'].values   # Z=0.0001 (z0001)
        ])
        self._chruslinska_Zs = jnp.array([0.03, 0.02, 0.01, 0.005, 0.001, 0.0001])

    def chruslinska_and_nelemans(
        self, z: jax.Array, metallicity: jax.Array | None
    ) -> jax.Array:
        """Calculate the Chruslinska & Nelemans (2019) tabulated SFRD.

        Args:
            z (jax.Array): The considered redshifts.
            metallicity (Optional[jax.Array]): The metallicity of the binary.

        Returns:
            jax.Array: The evaluated SFRD in Msol / yr / Mpc^3.
        """
        if metallicity is not None:
            metallicity_arr = jnp.atleast_1d(jnp.array(metallicity))
            z_arr = jnp.atleast_1d(jnp.array(z))

            if metallicity_arr.size == 1:
                m_val = metallicity_arr[0]
                idx = jnp.argmin(jnp.abs(self._chruslinska_Zs - m_val))
                row = self._chruslinska_sfrds[idx]
                return jnp.interp(z_arr, self._chruslinska_z, row)

            def get_single(single_z, single_m):
                idx = jnp.argmin(jnp.abs(self._chruslinska_Zs - single_m))
                return jnp.interp(single_z, self._chruslinska_z, self._chruslinska_sfrds[idx])

            return jax.vmap(get_single)(z_arr, metallicity_arr)
        else:
            interpolated_sfrds = jax.vmap(
                lambda row: jnp.interp(z, self._chruslinska_z, row)
            )(self._chruslinska_sfrds)
            return jnp.sum(interpolated_sfrds, axis=0)
