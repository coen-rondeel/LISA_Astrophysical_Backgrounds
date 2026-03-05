import yaml
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
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
    

def get_GWB_plot(frequencies: ArrayLike, omega_f: ArrayLike, save_path: Path):
    plt.loglog(frequencies, omega_f)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    