import jax
import jax.numpy as jnp
from lisatools.utils.constants import PI, G_SI, MSUN_SI, YRSID_SI, C_SI

RSUN_SI: float = 695700000.0 # nominal solar radius in meters (agrees with astropy and wolframalpha value)

KEPLER_CONST: float = (G_SI * MSUN_SI / (RSUN_SI**3))**(1/2) / (2.0 * PI)
GW_TIME_CONST: float = (96 / 5) * (2 * PI)**(8/3) * (G_SI * MSUN_SI)**(5/3) / (C_SI**5)
TAU_IN_MYR_CONST: float = 1 / (1e6 * YRSID_SI)

# TODO This file still contains my old docstring formatting 

@jax.jit
def chirp_mass(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """
    Calculates the chirp mass of a binary system.

    Parameters
    ----------
    m1 : jax.Array
        Mass of the first component. 
    m2 : jax.Array
        Mass of the second component. Must be in same unit as m1

    Returns
    -------
    jax.Array
        The chirp mass in same units as input.  
    """
    m_total: jax.Array = m1 + m2
    eta: jax.Array = (m1 * m2) / jnp.square(m_total)
    return m_total * jnp.power(eta, 0.6)


@jax.jit
def WD_radius(m: jax.Array) -> jax.Array:
    """
    Calculates the radius of a white dwarf.

    Parameters
    ----------
    m : jax.Array
        Mass of the white dwarf in solar masses. 

    Returns
    -------
    jax.Array
        The radius of the white dwarf in solar radii.  
    """
    C1: float = 1.44**(2/3)            
    C2: float = 1.44**(-2/3)          
    C3: float = 3.5 * (0.00057**(2/3)) 
    C4: float = 0.00057
    C5: float = 0.0114
    
    m_13 = jnp.cbrt(m)
    m_23: jax.Array = m_13 * m_13
    
    term1 = jnp.sqrt(C1 / m_23 - C2 * m_23)
    term2 = jnp.power(1.0 + (C3 / m_23) + (C4 / m), -2/3)
    
    return C5 * term1 * term2


@jax.jit
def a_min(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """
    Calculates minimal orbital separation between two white dwarfs.

    Parameters
    ----------
    m1 : jax.Array
        Mass of the white dwarf in solar masses. 
    m1 : jax.Array
        Mass of the other white dwarf in solar masses. 

    Returns
    -------
    jax.Array
        The minimal orbital separation in solar radii.  
    """
    r1: jax.Array = WD_radius(m1)
    r2: jax.Array = WD_radius(m2)
    
    q_13 = jnp.cbrt(m2) / jnp.cbrt(m1)    
    q_23: jax.Array = q_13 * q_13

    ap_min: jax.Array = r1 * (0.6 + q_23 * jnp.log1p(1 / q_13)) / 0.49
    as_min: jax.Array = r2 * (0.6 + 1/q_23 * jnp.log1p(q_13)) / 0.49

    return jnp.maximum(ap_min, as_min)

@jax.jit
def a_min_BHs(m1: jax.Array, m2: jax.Array) -> jax.Array:
    """
    Calculates minimal orbital separation between two black holes.

    Parameters
    ----------
    m1 : jax.Array
        Mass of the black hole in solar masses. 
    m1 : jax.Array
        Mass of the other black hole in solar masses. 

    Returns
    -------
    jax.Array
        The minimal orbital separation in solar radii.  
    """
    r_s1: jax.Array = 2 * G_SI * m1 * MSUN_SI / C_SI**2 / RSUN_SI
    r_s2: jax.Array = 2 * G_SI * m2 * MSUN_SI / C_SI**2 / RSUN_SI
    
    return r_s1 + r_s2



@jax.jit
def orbital_freq_kepler(m1: jax.Array, m2: jax.Array, a: jax.Array) -> jax.Array:
    """
    Calculates the Keplerian orbital frequency of a binary white dwarf

    Parameters
    ----------
    m1 : jax.Array
        Mass of the object primary in solar masses. 
    m2 : jax.Array
        Mass of the secondary in solar masses. 
    a: jax.Array
        The orbital separation between the binary components in solar radii.

    Returns
    -------
    jax.Array
        The Keplerian orbital frequency in Hz.  
    """
    a_3: jax.Array = a * a * a
    return KEPLER_CONST * jnp.sqrt((m1 + m2) / a_3)


@jax.jit
def K_factor(M_ch: jax.Array) -> jax.Array:
    """
    Calculates the GW frequency evolution factor of a circular binary system

    Parameters
    ----------
    M_ch : jax.Array
        The chirp mass of a binary system in solar masses.

    Returns
    -------
    jax.Array
        The GW frequency evolution factor K in s^(-5/3).  
    """
    M_ch_13 = jnp.cbrt(M_ch)
    M_ch_53: jax.Array = M_ch * M_ch_13 * M_ch_13
    return M_ch_53 * GW_TIME_CONST


@jax.jit
def tau_GW(f_start: jax.Array, f_end: jax.Array, K: jax.Array) -> jax.Array:
    """
    Calculates the characteristic timescale of a GW emission driven circular binary system to 
    evolve from a frequency f_start to a frequency f_end. 

    Parameters
    ----------
    f_start : jax.Array
        The considered starting frequency of the binary in Hz.
    f_end : jax.Array
        The considered end frequency of the binary in Hz. 
    K : jax.Array
        The GW frequency evolution factor, see function K_factor.

    Returns
    -------
    jax.Array
        The characteristic frequency evolution timescale in Myr.  
    """
    f_start_3: jax.Array = f_start * f_start * f_start
    f_start_83: jax.Array = f_start_3 / jnp.cbrt(f_start)

    f_end_3: jax.Array = f_end * f_end * f_end
    f_end_83: jax.Array = f_end_3 / jnp.cbrt(f_end)

    return (1/f_start_83 - 1/f_end_83) * (2.381 * TAU_IN_MYR_CONST / K)


@jax.jit
def orbital_freq_from_time(nu_start: jax.Array, evolve_time: jax.Array, K: jax.Array) -> jax.Array:
    """
    Calculates the orbital frequency of a GW emission driven circular binary system after evolving for a certain time.

    Parameters
    ----------
    nu_start : jax.Array
        The considered starting orbital frequency of the binary in Hz.
    evolve_time : jax.Array
        The time over which the binary evolves in Myr. 
    K : jax.Array
        The GW frequency evolution factor, see function K_factor.

    Returns
    -------
    jax.Array
        The orbital frequency of the binary after evolving for evolve_time.  
    """
    nu_start_83: jax.Array = (nu_start * nu_start * nu_start) / jnp.cbrt(nu_start)
    nu_end_inv_83: jax.Array = 1/nu_start_83 - (8 * evolve_time * K) / (3 * TAU_IN_MYR_CONST)
    nu_end_83: jax.Array = 1 / nu_end_inv_83
    nu_end_13 = jnp.sqrt(jnp.sqrt(jnp.sqrt(nu_end_83)))
    nu_end: jax.Array = nu_end_13 * nu_end_13 * nu_end_13
    return nu_end