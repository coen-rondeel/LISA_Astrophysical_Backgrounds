"""Initial Mass Function (IMF) frames and the conversions between them.

Every mass-normalized quantity in this package carries an implicit IMF. The
star formation rate density psi(z) is inferred from light dominated by massive
stars, so its normalization is set by whatever IMF the SFH paper calibrated
against; the star-forming mass M_SF behind a binary catalogue is set by the IMF
the population synthesis code sampled. The per-binary rate psi(z, Z) / M_SF(Z)
is only meaningful when both are expressed in the *same* frame.

The population is the target frame: the pair (N_catalogue, M_SF) is
self-consistent by construction and must not be rescaled, so psi is converted
into the population's frame instead. See ``dev/imf_handling_plan.md`` for the
full derivation.

References:
    Speagle et al. 2014, ApJS 214, 15 (doi:10.1088/0067-0049/214/2/15),
        Table C.2 and the indicator-dependent SFR conversions in Appendix C.
    Madau & Dickinson 2014, ARA&A 52, 415 (doi:10.1146/annurev-astro-081811-125615),
        section "The Stellar Initial Mass Function".

Reading Speagle's Table C.2 (this is easy to invert by accident): the table
lists ``psi | psi_Ch | 0.9434 psi_Ch | 0.5849 psi_Ch`` across the Chabrier /
Kroupa / Salpeter columns. Taken literally that places Salpeter *below*
Chabrier, which is backwards. Salpeter's excess of low-mass stars requires
more total mass per unit UV light. The factors bring the column's IMF *into*
the Chabrier frame. Three independent checks confirm that reading: Speagle's
own summary relation M*_K = 1.06 M*_C = 0.62 M*_S is its exact inverse;
0.9434 = 1/1.06 is the Kroupa -> Chabrier factor; and the dex column is
self-consistent (log10(0.9434) = -0.0253, log10(0.5849) = -0.2329).

Choice of the Salpeter entry (0.66, not Speagle's 0.62): Table C.2 applies the
same factors to M* and psi, but the Salpeter->Kroupa conversion depends on the
luminosity indicator that calibrated the quantity:

    ============================================  ==================
    basis                                         Salpeter -> Kroupa
    ============================================  ==================
    Table C.2 stellar-mass relation                     0.62  (M*)
    H-alpha recombination     (< 10 Myr)                0.68  (psi)
    FUV 1500 A                (~100 Myr)                0.63  (psi)
    NUV 2300 A                (~100 Myr)                0.64  (psi)
    total IR                  (~100 Myr)                0.86  (psi)
    MD14, FUV+IR                                        0.66  (psi)
    ============================================  ==================

The quantity converted here is a star formation rate density measured from
FUV+IR, so the MD14 value applies; 0.62 is the stellar-mass number and using it
for psi would apply a mass conversion to a rate. Combining 0.66 with Speagle's
Chabrier <-> Kroupa 1.06 reproduces every MD14 SFR conversion to within 1.2%
while keeping composition exact.
"""

# * Star formation rate conversions, pivoted on Kroupa (2001), 0.1-100 Msol:
# multiply a star formation rate expressed in frame X by TO_KROUPA[X] to bring
# it into the Kroupa frame. In dex: kroupa 0, chabrier +0.0253, salpeter
# -0.1805. These are *SFR* factors; stellar-mass conversions differ for
# Salpeter (0.62, Speagle Table C.2) and need their own table.
TO_KROUPA: dict[str, float] = {
    "kroupa": 1.0,
    "chabrier": 1.06,  # Speagle et al. 2014, Table C.2
    "salpeter": 0.66,  # MD14 FUV+IR; see the module docstring
}

# * Reference IMF of each SFH model, keyed by the config's SFH_name.
# Verified against the source papers (see dev/imf_handling_plan.md, section 2);
# the psi(z=0) prefactors corroborate the assignment, since MD14 quote 0.015 for
# Salpeter while both MF17 and Neijssel+2019 use 0.01 = 0.015 * 0.66 (Kroupa).
SFH_REFERENCE_IMF: dict[str, str] = {
    "madau_and_dickinson": "salpeter",  # MD14, stated
    "madau_and_fragos": "kroupa",  # MF17, Kroupa recalibration of MD14
    "strolger": "salpeter",  # Strolger et al. 2004
    "neijssel_2019": "kroupa",  # COMPAS samples Kroupa (2001)
    "chruslinska_and_nelemans": "kroupa",  # C&N 2019/2021, universal Kroupa
}


def normalize_imf_name(name: str) -> str:
    """Canonicalize an IMF name from the configuration file.

    Args:
        name (str): IMF name, case- and whitespace-insensitive.

    Returns:
        str: The canonical lower-case name, a key of :data:`TO_KROUPA`.

    Raises:
        ValueError: If the name is not a supported IMF.
    """
    canonical: str = str(name).strip().lower()

    if canonical not in TO_KROUPA:
        raise ValueError(
            f"Unknown IMF: {name!r}. Supported IMFs are {sorted(TO_KROUPA)}."
        )

    return canonical


def imf_sfr_factor(source: str, target: str) -> float:
    """Return the factor converting a star formation rate between IMF frames.

    ``psi_target = imf_sfr_factor(source, target) * psi_source``.

    Args:
        source (str): IMF the star formation rate is currently expressed in.
        target (str): IMF to express the star formation rate in.

    Returns:
        float: The multiplicative conversion factor, exactly 1.0 when the two
            frames are identical.
    """
    return TO_KROUPA[normalize_imf_name(source)] / TO_KROUPA[normalize_imf_name(target)]


def reference_imf_for_sfh(sfh_name: str) -> str | None:
    """Return the IMF frame a given SFH model is calibrated in.

    Args:
        sfh_name (str): The ``SFH_name`` from the configuration file.

    Returns:
        str | None: The canonical IMF name, or None for a model that is not in
            :data:`SFH_REFERENCE_IMF` (e.g. a user-added SFH method), so that
            the pipeline can degrade to no correction instead of failing.
    """
    return SFH_REFERENCE_IMF.get(sfh_name)
