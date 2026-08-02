"""Module containing the classes and models for Star Formation History (SFH)."""

import math
import warnings
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
from astropy.cosmology import Cosmology

from .imf import imf_sfr_factor, normalize_imf_name, reference_imf_for_sfh
from .utils import _where

# * Chruslinska & Nelemans (2019/2021) numerical SFRD data
# 12+log(O/H) grid used in their calculations (do not change)
_CHRUSLINSKA_FOH_MIN, _CHRUSLINSKA_FOH_MAX = 5.3, 9.7
_CHRUSLINSKA_N_FOH = 200
_CHRUSLINSKA_TIME_FILE = "Time_redshift_deltaT.dat"

# original download names from the Chruslinska & Nelemans online material:
#   MZ19 <- moderate_FOH_z_dM.dat
#   LZ19 <- low-Z_extreme_FOH_z_dM.dat
#   HZ19 <- high-Z_extreme_FOH_z_dM.dat
#   LZ21 <- 204f14SBBiC_FMR270_FOH_z_dM.dat
#   HZ21 <- 302f14SBBiC_FMR270_FOH_z_dM.dat
CHRUSLINSKA_MODELS: dict[str, str] = {
    "MZ19": "MZ19_FOH_z_dM.dat",
    "LZ19": "LZ19_FOH_z_dM.dat",
    "HZ19": "HZ19_FOH_z_dM.dat",
    "LZ21": "LZ21_FOH_z_dM.dat",
    "HZ21": "HZ21_FOH_z_dM.dat",
}

# {name: (Z_sun, 12+log(O/H)_sun)}
SOLAR_METALLICITY_SCALES: dict[str, tuple[float, float]] = {
    "Asplund09": (0.0134, 8.69),
    "AndersGrevesse89": (0.017, 8.83),
    "GrevesseSauval98": (0.0201, 8.93),
    "Villante14": (0.019, 8.85),
}


def _Z_to_foh(Z: jax.Array, solar_Z_scale: str) -> jax.Array:
    """Convert metallicity mass fraction Z to oxygen abundance 12+log(O/H)."""
    Z_sun, foh_sun = SOLAR_METALLICITY_SCALES[solar_Z_scale]
    return jnp.log10(jnp.asarray(Z) / Z_sun) + foh_sun


def _read_data_file(path: Path) -> jax.Array:
    """Read a whitespace-separated, ``#``-commented data file into an array.

    JAX has no file I/O, so the parsing is delegated to ``pandas``.

    Args:
        path (Path): Path to the data file.

    Returns:
        jax.Array: The file contents, shape (N_rows, N_columns).
    """
    return jnp.asarray(
        pd.read_csv(path, sep=r"\s+", comment="#", header=None).to_numpy()
    )


def _load_chruslinska_table(data_path: Path) -> dict[str, jax.Array]:
    """Load a Chruslinska & Nelemans SFRD table and its time grid.

    ``Time_redshift_deltaT.dat`` is expected in the same directory as the
    data file. The data tabulate the stellar mass (comoving, Msol / Mpc^3)
    formed per 12+log(O/H) bin (columns) and per redshift step (rows,
    z = 10 -> ~0).

    Args:
        data_path (Path): Path to a ``*FOH_z_dM*`` data file.

    Returns:
        dict: With keys (all redshift-ascending):
            ``z`` (N_z,): redshift of each row,
            ``sfrd_foh`` (N_z, 200): SFRD per FOH bin in Msol/yr/Mpc^3,
            ``total_sfrd`` (N_z,): SFRD summed over all FOH bins,
            ``foh_edges`` (201,): FOH bin edges (centers +/- half spacing).
    """
    time_table = _read_data_file(data_path.parent / _CHRUSLINSKA_TIME_FILE)
    redshift, delt = time_table[:, 1], time_table[:, 2]

    # a few entries are tiny negative numerical residues (~ -1e-7 Msol/Mpc^3)
    mass_per_bin = jnp.clip(_read_data_file(data_path), 0.0)  # (N_z, 200), Msol / Mpc^3

    if mass_per_bin.shape != (redshift.size, _CHRUSLINSKA_N_FOH):
        raise ValueError(
            f"Unexpected data shape {mass_per_bin.shape} in {data_path.name}; "
            f"expected ({redshift.size}, {_CHRUSLINSKA_N_FOH})."
        )

    sfrd_foh = mass_per_bin / (delt[:, None] * 1e6)  # Msol / yr / Mpc^3 per bin

    # flip rows so redshift is ascending (data start at z = 10)
    z = jnp.flip(redshift)
    sfrd_foh = jnp.flip(sfrd_foh, axis=0)

    foh_centers = jnp.linspace(
        _CHRUSLINSKA_FOH_MIN, _CHRUSLINSKA_FOH_MAX, _CHRUSLINSKA_N_FOH
    )
    d_foh = foh_centers[1] - foh_centers[0]
    foh_edges = jnp.concatenate(
        [foh_centers - d_foh / 2.0, foh_centers[-1:] + d_foh / 2.0]
    )

    return {
        "z": z,
        "sfrd_foh": sfrd_foh,
        "total_sfrd": sfrd_foh.sum(axis=1),
        "foh_edges": foh_edges,
    }


def _bin_chruslinska_sfrd(
    table: dict[str, jax.Array],
    Z_edges: jax.Array,
    coverage: str,
    solar_Z_scale: str,
) -> jax.Array:
    """Integrate the FOH distribution over each config metallicity bin.

    Args:
        table (dict): Output of :func:`_load_chruslinska_table`.
        Z_edges (jax.Array): Metallicity bin edges (mass fraction),
            ascending, length N_Zbins + 1.
        coverage (str): How to treat star formation outside the simulated
            metallicity range: ``"raw"`` drops it, ``"extend"`` stretches the
            outermost bin edges to the data limits, ``"renormalize"`` rescales
            each redshift row so the simulated bins carry the total SFRD
            (consistent with the Boileau et al. 2025 analytic convention).
        solar_Z_scale (str): Solar metallicity scale for the Z -> FOH
            conversion.

    Returns:
        jax.Array: SFRD per redshift and metallicity bin, shape
        (N_z, N_Zbins), in Msol / yr / Mpc^3.
    """
    foh_bin_edges = _Z_to_foh(jnp.sort(jnp.asarray(Z_edges)), solar_Z_scale)

    if coverage == "extend":
        foh_bin_edges = foh_bin_edges.at[0].set(
            jnp.minimum(foh_bin_edges[0], _CHRUSLINSKA_FOH_MIN - 1.0)
        )
        foh_bin_edges = foh_bin_edges.at[-1].set(
            jnp.maximum(foh_bin_edges[-1], _CHRUSLINSKA_FOH_MAX + 1.0)
        )
    elif coverage not in ("raw", "renormalize"):
        raise ValueError(
            f"Unknown SFH_coverage mode: {coverage!r}. "
            "Supported modes are 'raw', 'extend', 'renormalize'."
        )

    sfrd_foh = table["sfrd_foh"]  # (N_z, 200)

    # CDF of the SFRD along FOH at each redshift, defined on the bin edges
    cdf = jnp.concatenate(
        [jnp.zeros((sfrd_foh.shape[0], 1)), jnp.cumsum(sfrd_foh, axis=1)], axis=1
    )
    cdf_at_edges = jax.vmap(
        lambda cdf_row: jnp.interp(foh_bin_edges, table["foh_edges"], cdf_row)
    )(cdf)  # (N_z, N_Zbins + 1)
    sfrd_zZ = jnp.diff(cdf_at_edges, axis=1)  # (N_z, N_Zbins)

    if coverage == "renormalize":
        covered = sfrd_zZ.sum(axis=1)
        safe_covered = _where(covered > 0.0, covered, 1.0)
        scale = _where(covered > 0.0, table["total_sfrd"] / safe_covered, 0.0)
        sfrd_zZ = sfrd_zZ * scale[:, None]

    return sfrd_zZ


def _validated_imf_correction(correction: object) -> float:
    """Validate an explicit ``SFH_IMF_correction`` override from the config.

    Guards the two values that would otherwise silently zero the whole
    background: YAML parses ``false``/``no``/``off`` to a boolean and
    ``float(False)`` is 0.0, and a literal 0 is never a meaningful conversion.

    Args:
        correction (object): The raw value read from the configuration file.

    Returns:
        float: The validated multiplicative factor.

    Raises:
        ValueError: If the value is a boolean, not a number, or not a finite
            positive float.
    """
    if isinstance(correction, bool):
        raise ValueError(
            f"SFH_IMF_correction must be a positive number or 'auto', got the "
            f"boolean {correction!r}. Note that YAML reads false/no/off as a "
            "boolean; to disable the correction use 1.0, and to derive it from "
            "the IMF frames use 'auto'."
        )

    try:
        factor = float(correction)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"SFH_IMF_correction must be a positive number or 'auto', "
            f"got {correction!r}."
        ) from None

    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            f"SFH_IMF_correction must be a finite positive number, got {factor}. "
            "A value of 0 would silently zero the background; use 1.0 to apply "
            "no correction."
        )

    return factor


class StarFormationHistory:
    """SFH class instance.

    Note on IMF frames: the individual model methods (``madau_and_dickinson``,
    ``chruslinska_and_nelemans``, ...) always return psi in the **native frame
    of the published model**. The IMF correction into the population's frame is
    applied in exactly one place, at the single exit point of
    :meth:`delayed_SFH`, which is how all three GWB kernels evaluate the SFH.
    Adding the factor anywhere else would double-count it. See :mod:`.imf`.
    """

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

        self._resolve_imf_factor()

    def _resolve_imf_factor(self) -> None:
        """Determine the IMF correction applied to psi in :meth:`delayed_SFH`.

        The SFRD carries the IMF of the study it was calibrated against, while
        ``total_population_mass`` carries the IMF the population synthesis code
        sampled. The population is the target frame, so psi is converted into
        it (see :mod:`.imf`).

        Config keys:
            ``population_IMF`` (``population`` block): the frame of
                ``total_population_mass``. Omitting it disables the correction
                and emits a warning, which preserves the behaviour of configs
                written before this key existed.
            ``SFH_reference_IMF`` (optional, default None): overrides the
                registry lookup of the selected model's native frame.
            ``SFH_IMF_correction`` (optional, default ``"auto"``): a float
                bypasses the whole resolution and is used verbatim.

        Sets ``imf_factor``, ``imf_source`` and ``imf_target``.
        """
        sfh_config: dict = self._config["SFH"]
        population_config: dict = self._config.get("population", {})
        sfh_name: str = sfh_config["SFH_name"]

        correction = sfh_config.get("SFH_IMF_correction", "auto")
        if not (correction is None or correction == "auto"):
            self.imf_factor: float = _validated_imf_correction(correction)
            self.imf_source: str | None = None
            self.imf_target: str | None = None
            return

        source = sfh_config.get("SFH_reference_IMF") or reference_imf_for_sfh(sfh_name)
        target = population_config.get("population_IMF")

        if source is None or target is None:
            missing = "SFH_reference_IMF" if target is not None else "population_IMF"
            warnings.warn(
                f"No IMF correction applied to the {sfh_name} SFRD: {missing} is "
                "not set. The SFRD and total_population_mass are therefore "
                "assumed to already share an IMF frame. Set population_IMF in "
                "the population block (and SFH_reference_IMF for an SFH model "
                "that is not in the registry) to enable the correction.",
                stacklevel=3,
            )
            self.imf_factor = 1.0
            self.imf_source = source
            self.imf_target = target
            return

        self.imf_source = normalize_imf_name(source)
        self.imf_target = normalize_imf_name(target)
        self.imf_factor = imf_sfr_factor(self.imf_source, self.imf_target)

        print(
            f"Converting the {sfh_name} SFRD from its {self.imf_source} frame to "
            f"the {self.imf_target} frame of the population: "
            f"psi *= {self.imf_factor:.4f}"
        )

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

        This is the single point at which the IMF correction (``imf_factor``)
        is applied, so that every model reaches the GWB kernels in the IMF
        frame of the population. The model methods themselves return their
        native frame.

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
            psi: jax.Array = self._psi_function(
                age_at_star_formation, z_at_star_formation, metallicity
            )
        else:
            psi = self._psi_function(z_at_star_formation, metallicity)

        return psi * self.imf_factor

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
        """Load and preprocess the Chruslinska & Nelemans numerical SFRD data.

        Refs: https://doi.org/10.1093/mnras/stz2057 (2019) and
        https://doi.org/10.1093/mnras/stab2690 (2021).

        Config keys (``SFH`` block):
            ``SFH_path``: directory holding the data files (plus
                ``Time_redshift_deltaT.dat``), or a direct path to one
                ``*FOH_z_dM*`` file.
            ``SFRD_model``: model name (``MZ19``/``LZ19``/``HZ19``/``LZ21``/
                ``HZ21``), required when ``SFH_path`` is a directory.
            ``SFH_coverage`` (optional, default ``"renormalize"``): treatment
                of star formation outside the simulated metallicity range,
                see :func:`_bin_chruslinska_sfrd`.
            ``SFH_solar_Z_scale`` (optional, default ``"AndersGrevesse89"``):
                solar scale for the Z <-> 12+log(O/H) conversion.

        The tabulated data assume a Kroupa IMF. That is registered in
        :data:`.imf.SFH_REFERENCE_IMF` and handled by the shared correction in
        :meth:`delayed_SFH`; do not rescale the tables here.
        """
        sfh_config: dict = self._config["SFH"]

        data_path = Path(sfh_config["SFH_path"])

        if not data_path.exists():
            raise FileNotFoundError(
                f"SFH_path does not exist: {data_path}. Point it at the "
                f"directory holding the Chruslinska & Nelemans tables and "
                f"{_CHRUSLINSKA_TIME_FILE}, or at one *FOH_z_dM* file directly."
            )
        if data_path.is_dir():
            model_name: str = sfh_config["SFRD_model"]
            if model_name not in CHRUSLINSKA_MODELS:
                raise ValueError(
                    f"Unknown SFRD_model: {model_name!r}. Supported models "
                    f"are {sorted(CHRUSLINSKA_MODELS)}."
                )
            data_path = data_path / CHRUSLINSKA_MODELS[model_name]

        coverage: str = sfh_config.get("SFH_coverage", "renormalize")
        solar_Z_scale: str = sfh_config.get("SFH_solar_Z_scale", "AndersGrevesse89")

        table = _load_chruslinska_table(data_path)
        self._chr_z: jax.Array = jnp.array(table["z"])
        self._chr_z_index: jax.Array = jnp.arange(self._chr_z.size, dtype=float)
        self._chr_total_sfrd: jax.Array = jnp.array(table["total_sfrd"])

        if self._Z_edges is not None:
            self._chr_sfrd_zZ: jax.Array | None = jnp.array(
                _bin_chruslinska_sfrd(table, self._Z_edges, coverage, solar_Z_scale)
            )
        else:
            self._chr_sfrd_zZ = None

    def chruslinska_and_nelemans(
        self, redshifts: jax.Array, metallicity: jax.Array | None
    ) -> jax.Array:
        """Evaluate the Chruslinska & Nelemans numerical SFRD.

        The SFRD is zero outside the redshift range of the data (z > 10,
        before the onset of star formation in their calculation).

        Args:
            redshifts (jax.Array): Redshift at the moment of star formation.
            metallicity (jax.Array | None): The metallicity of each binary
                (snapped onto the config grid), or None for the total SFRD.

        Returns:
            jax.Array: The evaluated SFRD in Msol / yr / Mpc^3 (the per-bin
                fraction of the total when metallicity is provided).
        """
        if metallicity is None:
            return jnp.interp(redshifts, self._chr_z, self._chr_total_sfrd, right=0.0)

        sfrd_zZ = self._chr_sfrd_zZ
        Z_edges = self._Z_edges
        if sfrd_zZ is None or Z_edges is None:
            raise ValueError(
                "Multi-metallicity evaluation of the Chruslinska & Nelemans "
                "SFRD requires SFH_metallicity_bins in the config."
            )

        redshifts, metallicity = jnp.broadcast_arrays(redshifts, metallicity)

        Z_idx: jax.Array = jnp.clip(
            jnp.searchsorted(Z_edges, metallicity, side="right") - 1,
            0,
            sfrd_zZ.shape[1] - 1,
        )
        z_idx: jax.Array = jnp.interp(redshifts, self._chr_z, self._chr_z_index)

        psi: jax.Array = jax.scipy.ndimage.map_coordinates(
            sfrd_zZ,
            [z_idx, Z_idx.astype(float)],
            order=1,
            mode="nearest",
        )

        return _where(redshifts <= self._chr_z[-1], psi, 0.0)
