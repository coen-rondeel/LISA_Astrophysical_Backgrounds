import jax
import yaml
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

def get_config(path):
    """
    Read configuration parameter form file
    Args: 
        path -- Path to the yaml configuration file
    Return: 
        Dict with parameter
    """

    with open(path, 'r') as stream:
        return yaml.load(stream, yaml.FullLoader)
    

def get_GWB_plot(frequencies: jax.Array, omega_f: jax.Array, var_f: jax.Array, save_path: Path):
    plt.loglog(frequencies, omega_f, label="Expectation value")
    plt.fill_between(frequencies, omega_f - jnp.sqrt(var_f), omega_f + jnp.sqrt(var_f), alpha=0.35, label="1-sigma interval", color="tab:blue")
    plt.fill_between(frequencies, omega_f - 2.0*jnp.sqrt(var_f), omega_f + 2.0*jnp.sqrt(var_f), alpha=0.25, label="2-sigma interval", color="tab:blue")
    plt.fill_between(frequencies, omega_f - 3.0*jnp.sqrt(var_f), omega_f + 3.0*jnp.sqrt(var_f), alpha=0.15, label="3-sigma interval", color="tab:blue")
    plt.ylabel(r"GWB strength $\Omega_{\rm{GW}}(f)$")
    plt.xlabel(r"Frequency $f$")
    plt.legend()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    