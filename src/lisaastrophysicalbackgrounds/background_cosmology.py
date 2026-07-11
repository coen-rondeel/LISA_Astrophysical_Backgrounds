"""Background cosmology calculation and definitions."""

import astropy.cosmology as cosmo_module
import astropy.units as u
import jax
import jax.numpy as jnp
from astropy.cosmology import Cosmology


class BackgroundCosmology:
    """BackgroundCosmology class instance."""

    def __init__(self, config: dict) -> None:
        """Initialize the BC class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
        """
        self._config: dict = config

        self.get_cosmology()
        self.get_redshift_grid()

    def get_cosmology(self) -> None:
        """Get the configuration definied cosmology from astropy.cosmology."""
        if self._config["cosmology"]["standard"]:
            cosmo_name: str = self._config["cosmology"]["standard_cosmology"]

            with cosmo_module.default_cosmology.set(cosmo_name):
                cosmo = cosmo_module.default_cosmology.get()

            self.cosmo: Cosmology = cosmo

        else:
            cosmo_name: str = self._config["cosmology"]["custom_cosmology"][
                "cosmology_type"
            ]

            CosmoClass = getattr(cosmo_module, cosmo_name)

            cosmo = CosmoClass(
                **self._config["cosmology"]["custom_cosmology"]["params"]
            )

            self.cosmo: Cosmology = cosmo

    def get_redshift_grid(self) -> None:
        """Get the cosmological parameters using the config file.

        Raises:
            TypeError: An unsupported z_power_scale is defined in the config file.
            ValueError: An unsupported z_scale is defined in the config file.
        """
        self.z_min = float(self._config["cosmology"]["z_min"])
        self.z_max = float(self._config["cosmology"]["z_max"])
        self.N_zbins = int(float(self._config["cosmology"]["N_zbins"]))
        self.z_scale: str = self._config["cosmology"]["z_scale"]

        if self.z_scale == "linear":
            z_grid = jnp.linspace(self.z_min, self.z_max, 2 * self.N_zbins + 1)

        elif self.z_scale == "log10":
            z_grid = jnp.logspace(
                jnp.log10(self.z_min),
                jnp.log10(self.z_max),
                2 * self.N_zbins + 1,
                base=10,
            )

        elif self.z_scale == "log":
            z_grid = jnp.logspace(
                jnp.log(self.z_min),
                jnp.log(self.z_max),
                2 * self.N_zbins + 1,
                base=jnp.e,
            )

        elif self.z_scale == "power":
            self.z_power_scale: float = self._config["cosmology"]["z_power_scale"]

            if not (jnp.isnan(self.z_power_scale)):
                loga_factor = jnp.log(self.z_power_scale)

                z_grid = jnp.logspace(
                    jnp.log(self.z_min) / loga_factor,
                    jnp.log(self.z_max) / loga_factor,
                    2 * self.N_zbins + 1,
                    base=self.z_power_scale,
                )

            else:
                raise TypeError("Unsupported z_power_scale must be a number")

        elif self.z_scale == "lookback_time":
            # grid uniform in lookback time (matches Boileau et al. 2025).
            # Requires z_min > 0, as the lookback-time inversion is undefined at z=0.
            lookback_func = getattr(self.cosmo, "lookback_time")
            t_min: float = float(lookback_func(self.z_min).value)
            t_max: float = float(lookback_func(self.z_max).value)
            t_grid = jnp.linspace(t_min, t_max, 2 * self.N_zbins + 1)
            z_grid = jnp.array(
                [
                    float(cosmo_module.z_at_value(lookback_func, t * u.Gyr).value)
                    for t in t_grid
                ]
            )

        else:
            raise ValueError(f"Unsupported z_scale type: {self.z_scale}")

        self.z_grid: jax.Array = z_grid
        self.z_vals: jax.Array = z_grid[1::2]
        self.z_bins: jax.Array = z_grid[::2]

        comvingdistance_func = getattr(self.cosmo, "comoving_distance")
        self.DC_vals = jnp.asarray(comvingdistance_func(self.z_vals).value)  # in Mpc
        self.z_widths = jnp.diff(
            jnp.array(comvingdistance_func(self.z_bins).value)
        )  # in Mpc

        lookbacktime_func = getattr(self.cosmo, "lookback_time")
        lookback_times = (
            jnp.array(lookbacktime_func(self.z_vals).value) * 1000
        )  # in Myr
        self.z_time_since_z_max = (
            jnp.array(lookbacktime_func(self.z_max).value) * 1000 - lookback_times
        )  # in Myr

        age_func = getattr(self.cosmo, "age")
        self.ages = jnp.array(age_func(0).value) * 1000 - lookback_times  # in Myr
