"""Tests for the Chruslinska & Nelemans numerical SFRD in StarFormationHistory."""

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from astropy.cosmology import Planck18

from lisaastrophysicalbackgrounds.star_formation_history import (
    CHRUSLINSKA_MODELS,
    StarFormationHistory,
    _load_chruslinska_table,
)

# the package enables 64-bit globally (see gravitational_wave_background.py);
# mirror that here since this test imports star_formation_history directly
jax.config.update("jax_enable_x64", True)

SFRD_DIR = Path(__file__).resolve().parents[1] / "data" / "SFRDs"

# a small descending metallicity grid with geometric-midpoint edges
Z_CENTERS = [0.02, 0.005, 0.001, 0.0002]
Z_EDGES = [0.05, 0.01, 0.00224, 0.000448, 0.0001]


def make_sfh(
    multi_Z: bool, population_IMF: str | None = None, **sfh_extra
) -> StarFormationHistory:
    """Build a StarFormationHistory with the chruslinska_and_nelemans model."""
    config = {
        "population": {"population_IMF": population_IMF},
        "SFH": {
            "SFH_name": "chruslinska_and_nelemans",
            "SFH_path": str(SFRD_DIR),
            "SFRD_model": "MZ19",
            "SFH_metallicities": Z_CENTERS if multi_Z else 0.02,
            "SFH_metallicity_bins": Z_EDGES if multi_Z else None,
            **sfh_extra,
        },
    }
    # without population_IMF the un-corrected path warns; that warning is
    # asserted in test_missing_population_imf_warns_and_disables, so silence it
    # for the tests that are not about the IMF machinery
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No IMF correction")
        return StarFormationHistory(config, Planck18)


def test_all_chruslinska_models_load() -> None:
    """Every registered model file loads with the expected shape."""
    for name, filename in CHRUSLINSKA_MODELS.items():
        table = _load_chruslinska_table(SFRD_DIR / filename)
        assert table["sfrd_foh"].shape == (150, 200), name
        assert np.all(np.isfinite(table["sfrd_foh"])), name
        assert np.all(table["sfrd_foh"] >= 0.0), name
        assert np.all(np.diff(table["z"]) > 0), name


def test_single_Z_total_sfrd() -> None:
    """The single-metallicity path returns the total SFRD of the data."""
    sfh = make_sfh(multi_Z=False)
    table = _load_chruslinska_table(SFRD_DIR / CHRUSLINSKA_MODELS["MZ19"])

    psi = sfh.chruslinska_and_nelemans(jnp.array(table["z"]), None)
    np.testing.assert_allclose(np.asarray(psi), table["total_sfrd"], rtol=1e-12)

    # zero before the onset of star formation in the data (z > 10)
    assert float(sfh.chruslinska_and_nelemans(jnp.array([10.5]), None)[0]) == 0.0

    # plausible cosmic-SFH amplitude and peak location
    total = table["total_sfrd"]
    assert 0.01 < total.max() < 1.0
    assert 1.0 < table["z"][total.argmax()] < 3.0


def test_multi_Z_mass_conservation() -> None:
    """Summing psi over the metallicity grid recovers the total SFRD."""
    sfh = make_sfh(multi_Z=True)
    z_eval = jnp.linspace(0.05, 9.0, 64)

    psi_per_Z = jax.vmap(
        lambda Z: sfh.chruslinska_and_nelemans(z_eval, jnp.full(z_eval.shape, Z))
    )(jnp.array(Z_CENTERS))
    total = sfh.chruslinska_and_nelemans(z_eval, None)

    np.testing.assert_allclose(
        np.asarray(psi_per_Z.sum(axis=0)), np.asarray(total), rtol=5e-3
    )


def test_multi_Z_jit_and_broadcasting() -> None:
    """The psi function works under jit with per-binary arrays and scalar z."""
    sfh = make_sfh(multi_Z=True)
    rng = np.random.default_rng(0)
    N = 5000
    Z_binaries = jnp.array(rng.choice(Z_CENTERS, size=N))
    z_binaries = jnp.array(rng.uniform(0.0, 11.0, size=N))

    psi = jax.jit(sfh.chruslinska_and_nelemans)(z_binaries, Z_binaries)
    assert psi.shape == (N,)
    assert bool(jnp.all(jnp.isfinite(psi)))
    assert bool(jnp.all(psi >= 0.0))

    # scalar redshift broadcast against per-binary metallicities
    psi_scalar = jax.jit(sfh.chruslinska_and_nelemans)(jnp.float64(1.5), Z_binaries)
    assert psi_scalar.shape == (N,)


def test_delayed_SFH_dispatch() -> None:
    """delayed_SFH dispatches to the numerical SFRD with zero delay."""
    sfh = make_sfh(multi_Z=True)
    z_ref = 1.0
    age = jnp.array(Planck18.age(z_ref).value * 1000)  # Myr

    psi = sfh.delayed_SFH(age, jnp.zeros(4), jnp.array(Z_CENTERS))
    expected = sfh.chruslinska_and_nelemans(jnp.full(4, z_ref), jnp.array(Z_CENTERS))
    np.testing.assert_allclose(np.asarray(psi), np.asarray(expected), rtol=1e-3)


def test_coverage_modes() -> None:
    """Raw coverage captures less star formation than renormalize."""
    z_eval = jnp.linspace(0.05, 9.0, 32)

    def summed_psi(sfh: StarFormationHistory) -> np.ndarray:
        per_Z = jax.vmap(
            lambda Z: sfh.chruslinska_and_nelemans(z_eval, jnp.full(z_eval.shape, Z))
        )(jnp.array(Z_CENTERS))
        return np.asarray(per_Z.sum(axis=0))

    raw = summed_psi(make_sfh(multi_Z=True, SFH_coverage="raw"))
    renorm = summed_psi(make_sfh(multi_Z=True, SFH_coverage="renormalize"))
    total = np.asarray(make_sfh(multi_Z=True).chruslinska_and_nelemans(z_eval, None))

    assert np.all(raw <= total * (1 + 1e-12))
    np.testing.assert_allclose(renorm, total, rtol=5e-3)


def _delayed_at(sfh: StarFormationHistory, z_ref: float, Z: float) -> np.ndarray:
    """Evaluate delayed_SFH with zero delay at a single redshift."""
    age = jnp.array(Planck18.age(z_ref).value * 1000)  # Myr
    return np.asarray(sfh.delayed_SFH(age, jnp.zeros(1), jnp.array([Z])))


def test_imf_correction_scales_linearly() -> None:
    """SFH_IMF_correction rescales the SFRD delivered by delayed_SFH."""
    base = make_sfh(multi_Z=True, SFH_IMF_correction=1.0)
    conv = make_sfh(multi_Z=True, SFH_IMF_correction=1 / 1.06)

    np.testing.assert_allclose(
        _delayed_at(conv, 1.0, Z_CENTERS[1]),
        _delayed_at(base, 1.0, Z_CENTERS[1]) / 1.06,
        rtol=1e-12,
    )


def test_model_methods_return_native_frame() -> None:
    """The model methods are un-corrected; only delayed_SFH applies the factor."""
    sfh = make_sfh(multi_Z=True, population_IMF="chabrier")
    assert sfh.imf_factor == pytest.approx(1 / 1.06)

    z = jnp.linspace(0.1, 5.0, 16)
    Z = jnp.full(z.shape, Z_CENTERS[1])
    native = make_sfh(multi_Z=True, SFH_IMF_correction=1.0).chruslinska_and_nelemans(
        z, Z
    )
    np.testing.assert_allclose(
        np.asarray(sfh.chruslinska_and_nelemans(z, Z)), np.asarray(native), rtol=1e-12
    )
    np.testing.assert_allclose(
        _delayed_at(sfh, 1.0, Z_CENTERS[1]),
        _delayed_at(make_sfh(multi_Z=True, SFH_IMF_correction=1.0), 1.0, Z_CENTERS[1])
        / 1.06,
        rtol=1e-12,
    )


def test_auto_factor_from_imf_frames() -> None:
    """The numerical SFRD is Kroupa, so a Kroupa population needs no correction."""
    assert make_sfh(multi_Z=True, population_IMF="kroupa").imf_factor == 1.0
    assert make_sfh(
        multi_Z=True, population_IMF="salpeter"
    ).imf_factor == pytest.approx(1 / 0.66)


def test_float_override_beats_auto() -> None:
    """An explicit float bypasses the IMF frames entirely."""
    sfh = make_sfh(multi_Z=True, population_IMF="salpeter", SFH_IMF_correction=0.5)
    assert sfh.imf_factor == 0.5
    assert sfh.imf_source is None and sfh.imf_target is None


@pytest.mark.parametrize(
    ("correction", "match"),
    [
        (False, "boolean"),  # YAML reads false/no/off as a boolean
        (0, "finite positive"),  # would silently zero the whole background
        (-0.5, "finite positive"),
        ("not_a_number", "positive number or 'auto'"),
    ],
)
def test_invalid_imf_correction_raises(correction: object, match: str) -> None:
    """Values that would silently zero or break the SFRD are rejected."""
    with pytest.raises(ValueError, match=match):
        make_sfh(multi_Z=True, population_IMF="kroupa", SFH_IMF_correction=correction)


def test_missing_population_imf_warns_and_disables() -> None:
    """Without population_IMF the SFRD is left in its native frame."""
    with pytest.warns(UserWarning, match="population_IMF"):
        sfh = StarFormationHistory(
            {
                "SFH": {
                    "SFH_name": "chruslinska_and_nelemans",
                    "SFH_path": str(SFRD_DIR),
                    "SFRD_model": "MZ19",
                    "SFH_metallicities": 0.02,
                    "SFH_metallicity_bins": None,
                }
            },
            Planck18,
        )
    assert sfh.imf_factor == 1.0


def test_analytic_models_are_corrected() -> None:
    """The correction reaches the analytic SFHs, not just the numerical one."""
    config = {
        "population": {"population_IMF": "kroupa"},
        "SFH": {
            "SFH_name": "madau_and_dickinson",
            "SFH_metallicities": 0.02,
            "SFH_metallicity_bins": None,
        },
    }
    sfh = StarFormationHistory(config, Planck18)
    assert sfh.imf_factor == pytest.approx(0.66)

    age = jnp.array(Planck18.age(1.0).value * 1000)
    delayed = np.asarray(sfh.delayed_SFH(age, jnp.zeros(1), None))
    native = np.asarray(sfh.madau_and_dickinson(jnp.array([1.0]), None))
    np.testing.assert_allclose(delayed, native * 0.66, rtol=1e-6)


def test_metallicity_distribution_is_imf_invariant() -> None:
    """dP/dZ is renormalized over the grid, so the IMF factor cancels."""
    z = jnp.linspace(0.1, 5.0, 8)
    Z = jnp.full(z.shape, Z_CENTERS[1])
    kroupa = make_sfh(multi_Z=True, population_IMF="kroupa")
    salpeter = make_sfh(multi_Z=True, population_IMF="salpeter")
    np.testing.assert_allclose(
        np.asarray(kroupa.metallicity_distribution(z, Z)),
        np.asarray(salpeter.metallicity_distribution(z, Z)),
        rtol=1e-12,
    )


def test_invalid_config_raises() -> None:
    """Unknown model names and coverage modes raise informative errors."""
    with pytest.raises(ValueError, match="SFRD_model"):
        make_sfh(multi_Z=False, SFRD_model="bogus")
    with pytest.raises(ValueError, match="SFH_coverage"):
        make_sfh(multi_Z=True, SFH_coverage="bogus")
