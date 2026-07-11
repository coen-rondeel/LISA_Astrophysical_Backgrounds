"""Module for importing and preprocessing binary populations for GWB calculations."""

from typing import cast

import jax
import jax.numpy as jnp
import pandas as pd
from numpy.typing import ArrayLike
from pandas import DataFrame

from .physics import (
    K_factor,
    a_min,
    a_min_BHs,
    chirp_mass,
    orbital_freq_kepler,
    tau_GW,
)
from .utils import load_T0_data, select_evolutionary_states


class PreprocessPopulation:
    """The PreprocessPopulation instance."""

    def __init__(
        self,
        config: dict,
        t0: ArrayLike | None = None,
        m1: ArrayLike | None = None,
        m2: ArrayLike | None = None,
        a: ArrayLike | None = None,
        Z: ArrayLike | None = None,
    ) -> None:
        """Initialize the PC class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
            t0 (ArrayLike | None): Initial birth times.
            m1 (ArrayLike | None): Primary masses.
            m2 (ArrayLike | None): Secondary masses.
            a (ArrayLike | None): Semi-major axes.
            Z (ArrayLike | None): Metallicities.
        """
        self._config: dict = config

        tot_pop_mass = self._config["population"]["total_population_mass"]
        if isinstance(tot_pop_mass, (list, tuple)):
            self.total_population_mass = jnp.array(tot_pop_mass, dtype=jnp.float64)
        else:
            self.total_population_mass = float(tot_pop_mass)

        self.t0: jax.Array = (
            jnp.asarray(t0) if t0 is not None else cast(jax.Array, None)
        )
        self.m1: jax.Array = (
            jnp.asarray(m1) if m1 is not None else cast(jax.Array, None)
        )
        self.m2: jax.Array = (
            jnp.asarray(m2) if m2 is not None else cast(jax.Array, None)
        )
        self.a: jax.Array = jnp.asarray(a) if a is not None else cast(jax.Array, None)
        self.Z = jnp.asarray(Z) if Z is not None else None

        self.import_population()

        self.check_parameters()

        # check dimensionality for multiple masses vs metallicities
        unique_Z = jnp.unique(self.Z) if self.Z is not None else jnp.array([0.0])
        if isinstance(self.total_population_mass, jax.Array):
            if self.total_population_mass.size != unique_Z.size:
                if self.total_population_mass.size > unique_Z.size:
                    raise ValueError(
                        f"Listed total population masses "
                        f"({self.total_population_mass.size}) is larger than "
                        f"unique metallicities ({unique_Z.size})."
                    )
                else:
                    raise ValueError(
                        f"Listed total population masses "
                        f"({self.total_population_mass.size}) must equal the "
                        f"amount of unique metallicities ({unique_Z.size})."
                    )

        print(
            f"The metallicities of this population are "
            f"{self.Z if self.Z is None else jnp.unique(self.Z)}"
        )

    def import_population(self) -> None:
        """Import the binary population using specified configuration routines."""
        population_import_name = self._config["population"]["population_import_name"]
        if population_import_name is not None:
            import_func = getattr(self, population_import_name)
            import_func()
        else:
            print("No population importer was defined in the config file.")
            print(
                "To continue with manual imports:\n"
                "  Set GravitationalWaveBackground.population = "
                "PreprocessPopulation(t0, m1, m2, a, Z)\n"
                "  Run GravitationalWaveBackground.clean_population()"
            )
            raise ValueError("Population import and pre-processing was stopped")

    def check_parameters(self) -> None:
        """Initialize parameters if missing for circular/quadrupolar WD binaries."""
        # TODO do this for other types?
        gwb_input_params = [
            "t0",
            "M_ch",
            "M_ch_pow",
            "K_factor",
            "nu0",
            "numax",
            "merger_time",
        ]

        for param_name in gwb_input_params:
            current_param = getattr(self, param_name, None)
            if current_param is None:
                get_current_param = getattr(self, "get_" + param_name)
                get_current_param()

    def BinCodex_importer(self) -> None:
        """Import population data using the BinCodex framework format."""
        bin_path = self._config["population"]["population_path"]
        code_name = self._config["population"]["population_synthesis_code_name"]

        #! Current limitation here is that it can only handle 1 metallicity
        metallicity = self._config["SFH"]["SFH_metallicities"]
        assert len(metallicity) == 1, (
            "The T0 data loader can only handle a single metallicity. "
            "Needs future proofing."
        )

        data_T0, _ = load_T0_data(bin_path, code=code_name, metallicity=metallicity)

        ZAMS, WDMS, DWD = select_evolutionary_states(d=data_T0)  # type: ignore
        del (
            ZAMS,
            WDMS,
        )  # ? maybe we can do something interesting with this data as well?

        self.t0 = jnp.asarray(DWD.time)
        self.a = jnp.asarray(DWD.semiMajor)
        self.m1 = jnp.asarray(DWD.mass1)
        self.m2 = jnp.asarray(DWD.mass2)
        self.Z = metallicity * jnp.ones_like(self.t0)

    def simple_single_Z_import(self) -> None:
        """Import a galaxy population of BWDs containing only one metallicity."""
        population_df = pd.read_csv(self._config["population"]["population_path"])
        population_df: DataFrame = population_df.apply(
            pd.to_numeric, errors="coerce"
        )  # 'coerce' will turn invalid parsing into NaN

        self.t0 = jnp.array(population_df["t0"].values)

        self.M_ch: jax.Array = chirp_mass(
            jnp.array(population_df["m1"].values), jnp.array(population_df["m2"].values)
        )

        self.M_ch_pow = jnp.cbrt(self.M_ch**5)

        self.K_factor: jax.Array = K_factor(self.M_ch)

        self.nu0: jax.Array = orbital_freq_kepler(
            jnp.array(population_df["m1"].values),
            jnp.array(population_df["m2"].values),
            jnp.array(population_df["a"].values),
        )

        self.numax: jax.Array = orbital_freq_kepler(
            jnp.array(population_df["m1"].values),
            jnp.array(population_df["m2"].values),
            a_min(
                jnp.array(population_df["m1"].values),
                jnp.array(population_df["m2"].values),
            ),
        )

        self.merger_time: jax.Array = tau_GW(
            2 * self.nu0, 2 * self.numax, self.K_factor
        )

    def simple_multi_Z_import(self) -> None:
        """Import a galaxy population of BWDs containing multiple metallicities."""
        population_df = pd.read_csv(self._config["population"]["population_path"])
        population_df: DataFrame = population_df.apply(
            pd.to_numeric, errors="coerce"
        )  # 'coerce' will turn invalid parsing into NaN

        self.t0 = jnp.array(population_df["t0"].values)
        self.m1 = jnp.array(population_df["m1"].values)
        self.m2 = jnp.array(population_df["m2"].values)
        self.a = jnp.array(population_df["a"].values)

        try:
            self.Z = jnp.array(population_df["Z"].values)
        except KeyError:
            raise KeyError(
                "The population import should provide a metallicity column "
                "named 'Z' for each binary."
            )

    def mock_BH_import(self) -> None:
        """Import a galaxy population of BBHs containing only one metallicity."""
        population_df = pd.read_csv(
            self._config["population"]["population_path"],
            usecols=["t0", "a", "m1", "m2"],
        )

        self.t0 = jnp.array(population_df["t0"].values)

        self.M_ch: jax.Array = chirp_mass(
            jnp.array(population_df["m1"].values), jnp.array(population_df["m2"].values)
        )

        self.M_ch_pow = jnp.cbrt(self.M_ch**5)

        self.K_factor: jax.Array = K_factor(self.M_ch)

        self.nu0: jax.Array = orbital_freq_kepler(
            jnp.array(population_df["m1"].values),
            jnp.array(population_df["m2"].values),
            jnp.array(population_df["a"].values),
        )

        self.numax: jax.Array = orbital_freq_kepler(
            jnp.array(population_df["m1"].values),
            jnp.array(population_df["m2"].values),
            a_min_BHs(
                jnp.array(population_df["m1"].values),
                jnp.array(population_df["m2"].values),
            ),
        )

        self.merger_time: jax.Array = tau_GW(
            2 * self.nu0, 2 * self.numax, self.K_factor
        )

    # --- Fallback getters for incomplete configuration imports ---

    def get_t0(self) -> ValueError:
        """Raise error when population birth time is missing."""
        raise ValueError(
            "The time of birth of the binary was not correctly imported. "
            "Please ensure that t0 is included in the population and import function"
        )

    def get_M_ch(self) -> None:
        """Calculate the chirp mass attribute from available components."""
        try:
            M_ch_pow = getattr(self, "M_ch_pow")
        except AttributeError:
            M_ch_pow = None

        if self.m1 is None or self.m2 is None and M_ch_pow is None:
            raise ValueError(
                "It appears that the mass related imports of the populations "
                "where unsuccessful. Please ensure that the population has mass "
                "attributes"
            )
        elif M_ch_pow is not None:
            self.M_ch = jnp.power(jnp.array(self.M_ch_pow), 3 / 5)
        else:
            self.M_ch = chirp_mass(self.m1, self.m2)

    def get_M_ch_pow(self) -> None:
        """Calculate the chirp mass power factor to the 5/3rd power equivalent."""
        self.M_ch_pow = jnp.cbrt(self.M_ch**5)

    def get_K_factor(self) -> None:
        """Evaluate the constant gravitational radiation damping scale factor."""
        self.K_factor = K_factor(self.M_ch)

    def get_nu0(self) -> None:
        """Determine initial orbital frequency via Kepler's laws."""
        if self.m1 is None or self.m2 is None or self.a is None:
            raise ValueError(
                "The population import should either provide m1, m2, and a, OR nu0."
            )
        else:
            self.nu0 = orbital_freq_kepler(self.m1, self.m2, self.a)

    def get_numax(self) -> None:
        """Evaluate maximum orbital target frequency boundary limit configurations."""
        if self.m1 is None or self.m2 is None or self.a is None:
            raise ValueError(
                "The population import should either provide m1 and m2, OR numax."
            )
        else:
            print("We are assuming that the binaries are DWDs.")
            self.numax = orbital_freq_kepler(self.m1, self.m2, a_min(self.m1, self.m2))

    def get_merger_time(self) -> None:
        """Calculate remaining orbital lifespan profile duration."""
        self.merger_time = tau_GW(2 * self.nu0, 2 * self.numax, self.K_factor)


if __name__ == "__main__":
    # run with python -m src.lisaastrophysicalbackgrounds.preprocess_population
    jax.config.update("jax_enable_x64", True)
    base_data_path = (
        "/Users/rrondeel/Code/LISA_data_analysis/"
        "LISA_Astrophysical_Backgrounds/data/populations/"
        "Initials_SeBa_Gamma175Alpha4_Z02.txt"
    )

    proc_cat = PreprocessPopulation(
        config={
            "population": {
                "population_import_name": "simple_multi_Z_import",
                "total_population_mass": 42.0,
                "population_path": base_data_path,
            }
        }
    )
    print("starting on second population")
    proc_cat_2 = PreprocessPopulation(
        config={
            "population": {
                "population_import_name": None,
                "total_population_mass": 42.0,
                "population_path": base_data_path,
            }
        },
        t0=[1000, 2000],
        m1=[0.6, 0.8],
        m2=[0.5, 0.7],
        a=[5, 7],
    )
    breakpoint()
