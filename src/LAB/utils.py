import yaml
import matplotlib.pyplot as plt

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
    

def get_GWB_plot(frequencies, omega_f, save_path):
    raise NotImplementedError("The plot function still needs to be implemented")