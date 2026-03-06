# tests/test_physics.py

import jax.numpy as jnp
import numpy as np
from lisaastrophysicalbackgrounds.physics import (
    chirp_mass, orbital_freq_kepler, tau_GW
)

def test_chirp_mass() -> None:
    """Test the chirp mass calculation for symmetry and positive output."""
    m1 = jnp.array([1.0, 2.0])
    m2 = jnp.array([1.0, 3.0])
    
    m_ch = chirp_mass(m1, m2)
    
    assert m_ch.shape == (2,)
    np.testing.assert_allclose(m_ch, chirp_mass(m2, m1), rtol=1e-6)
    assert jnp.all(m_ch > 0)


def test_orbital_freq_kepler() -> None:
    """Test orbital frequency behavior (larger separation -> lower frequency)."""
    m1 = jnp.array([1.0])
    m2 = jnp.array([1.0])
    
    a_small = jnp.array([0.1])
    a_large = jnp.array([1.0])
    
    freq_small = orbital_freq_kepler(m1, m2, a_small)
    freq_large = orbital_freq_kepler(m1, m2, a_large)
    
    assert freq_small[0] > freq_large[0], "Smaller separation should yield higher frequency"


def test_tau_gw_positive() -> None:
    """Test that characteristic timescale is positive for increasing frequency."""
    f_start = jnp.array([1e-4])
    f_end = jnp.array([1e-3])
    K = jnp.array([1e-10])
    
    tau = tau_GW(f_start, f_end, K)
    assert tau[0] > 0, "Time to evolve to a higher frequency must be positive"