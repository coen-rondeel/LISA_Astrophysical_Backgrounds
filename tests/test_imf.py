"""Tests for the IMF frames and conversion factors."""

import itertools

import pytest

from lisaastrophysicalbackgrounds.imf import (
    SFH_REFERENCE_IMF,
    TO_KROUPA,
    imf_sfr_factor,
    normalize_imf_name,
    reference_imf_for_sfh,
)
from lisaastrophysicalbackgrounds.star_formation_history import StarFormationHistory

IMFS = sorted(TO_KROUPA)

# independently quoted literature values (Madau & Dickinson 2014); the adopted
# table must reproduce these without being derived from them
LITERATURE = {
    ("salpeter", "kroupa"): 0.66,
    ("chabrier", "kroupa"): 1.06,
    ("salpeter", "chabrier"): 0.63,
    ("kroupa", "chabrier"): 0.94,
    ("kroupa", "salpeter"): 1.50,
    ("chabrier", "salpeter"): 1.59,
}


def test_identity_is_exactly_one() -> None:
    """Converting within one frame must not perturb the SFRD at all."""
    for imf in IMFS:
        assert imf_sfr_factor(imf, imf) == 1.0


def test_factors_compose() -> None:
    """f(A->B) * f(B->C) == f(A->C) for every triple."""
    for a, b, c in itertools.product(IMFS, repeat=3):
        assert imf_sfr_factor(a, b) * imf_sfr_factor(b, c) == pytest.approx(
            imf_sfr_factor(a, c), rel=1e-12
        )


def test_round_trip_is_unity() -> None:
    """f(A->B) is the exact inverse of f(B->A)."""
    for a, b in itertools.permutations(IMFS, 2):
        assert imf_sfr_factor(a, b) * imf_sfr_factor(b, a) == pytest.approx(1.0)


def test_matches_literature_values() -> None:
    """Every conversion agrees with the published value to better than 2%."""
    for (source, target), expected in LITERATURE.items():
        assert imf_sfr_factor(source, target) == pytest.approx(expected, rel=0.02)


def test_salpeter_sfr_exceeds_kroupa_and_chabrier() -> None:
    """Salpeter's excess of low-mass stars means more mass per unit UV light.

    This is the sanity check that catches an inverted reading of the Speagle
    conversion table.
    """
    assert imf_sfr_factor("salpeter", "kroupa") < 1.0
    assert imf_sfr_factor("salpeter", "chabrier") < 1.0
    assert imf_sfr_factor("chabrier", "kroupa") > 1.0


def test_name_normalization() -> None:
    """IMF names are case- and whitespace-insensitive."""
    assert normalize_imf_name("  KrOuPa ") == "kroupa"
    assert imf_sfr_factor("Salpeter", "KROUPA") == imf_sfr_factor("salpeter", "kroupa")


def test_unknown_imf_raises() -> None:
    """An unsupported IMF name reports the supported set."""
    with pytest.raises(ValueError, match="chabrier"):
        normalize_imf_name("miller_scalo")


def test_registry_matches_sfh_methods() -> None:
    """Every registered SFH name is an actual model method, and vice versa."""
    for sfh_name, imf in SFH_REFERENCE_IMF.items():
        assert callable(getattr(StarFormationHistory, sfh_name, None)), sfh_name
        assert imf in TO_KROUPA, sfh_name

    documented = set(SFH_REFERENCE_IMF)
    implemented = {
        "madau_and_dickinson",
        "madau_and_fragos",
        "strolger",
        "neijssel_2019",
        "chruslinska_and_nelemans",
    }
    assert documented == implemented


def test_unregistered_sfh_returns_none() -> None:
    """A user-added SFH degrades to no correction rather than crashing."""
    assert reference_imf_for_sfh("my_custom_sfh") is None
