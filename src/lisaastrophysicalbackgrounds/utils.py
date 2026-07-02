import jax
import yaml
import jax.numpy as jnp
import pandas as pd
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
    
    
    
#* ========================== Helper functions to import the BinCodex format ==========================
"""
These functions are directly taken from the Synthetic UCB Catalogs project (github.com/Synthetic-UCB-Catalogs).
Therefore, all credits for writing these functions go to the members of that project. 
These functions are merely copied from the 'analysis-scripts' sub-repository to make this package as light weight as possible.
At the time of writing this part of the code (June 2026), there was no pip-installable version of some package that contained
the relevant code below.
"""

# main importer

def load_T0_data(ifilepath, code=None, **kwargs):
    """Read in standardized output Common Core data and select at DWD formation

    Note: all codes should save their T0 dataframes as hdf, for speed and storage
    
    Parameters
    ----------
    ifilepath : `str`
        ifilepath to T0 datafile 

    code : 
        name of code (only required for non-rapid codes, ComBinE and SEVN)

    **kwargs
        metallicty : `float`
            metallicity of the data if code=='SEVN'; this is usually encoded in the path

    Returns
    -------
    dat : `pandas.DataFrame`
        all data in T0 format
        
    header : `pandas.DataFrame`
        header for dat
    """
    if code == "ComBinE_csv":
        col_standard = ["ID","UID","SID","time","event",
                        "semiMajor","eccentricity","type1",
                        "mass1","radius1","Teff1","massHeCore1",
                        "type2","mass2","radius2","Teff2","massHeCore2",
                        "envBindEn","massCOCore1","massCOCore2",
                        "radiusRL1","radiusRL2","period",
                        "luminosity1","luminosity2"]

        # load the data
        dat = pd.read_csv(ifilepath, skiprows=6, names=col_standard)
        lines_number = 6
        with open(ifilepath) as input_file:
            head = [next(input_file) for _ in range(lines_number)]
            T0_info = head[4].replace(" ", "").split(",")
        
            header_info = {"cofVer" : float(T0_info[0]), 
                           "cofLevel": T0_info[1],
                           "cofExtension": "None", 
                           "bpsName": T0_info[3],
                           "bpsVer": T0_info[4], 
                           "contact": T0_info[5], 
                           "NSYS": int(T0_info[6]), 
                           "NLINES": int(T0_info[7]),
                           "Z": float(T0_info[8].replace("\n",""))}
    elif code == "SEVN":
        metallicity = kwargs.pop('metallicity')
        col_standard = ["ID","UID","time","event","semiMajor","eccentricity",
                        "type1","mass1","radius1","Teff1","massHecore1",
                        "type2","mass2","radius2","Teff2","massHecore2"]
        #read in the data with the columns
        dat = pd.read_csv(ifilepath, skiprows=3, names=col_standard)

        #read in the T0 info in the header
        lines_number = 3
        with open(ifilepath) as input_file:
            head = [next(input_file) for _ in range(lines_number)]
            T0_info = head[1].replace(" ", "").split(",")
            header_info = {"cofVer" : float(T0_info[0]), 
                           "cofLevel": T0_info[1],
                           "cofExtension": "None", 
                           "bpsName": T0_info[3],
                           "bpsVer": T0_info[4], 
                           "contact": T0_info[5], 
                           "NSYS": int(T0_info[6]), 
                           "NLINES": int(T0_info[7]),
                           "Z": metallicity}
    elif code == "BPASS":
        metallicity = 0.02
        #read in the column names from the third line of the file
        with open(ifilepath) as f:
            for _ in range(2):
                next(f)
            col_names = next(f).strip().split(',')
        #read in the data with the columns
        dat = pd.read_csv(ifilepath, sep='\s+', skiprows=3, names=col_names, skip_blank_lines=True)
        
        ##read in the data with the columns
        #dat = pd.read_csv(ifilepath, sep='\s+', skiprows=2, skip_blank_lines=True)
        
        #read in the T0 info in the header
        lines_number = 2
        with open(ifilepath) as input_file:
            head = [next(input_file) for _ in range(lines_number)]
            T0_info = head[1].replace(" ", "").split(",")
            header_info = {"cofVer" : float(T0_info[0]), 
                           "cofLevel": T0_info[1],
                           "cofExtension": "None", 
                           "bpsName": T0_info[3],
                           "bpsVer": T0_info[4], 
                           "contact": T0_info[5], 
                           "NSYS": int(T0_info[6]), 
                           "NLINES": int(T0_info[7]),
                           "Z": metallicity}

    #elif code in ["COMPAS", "COSMIC", "SeBa", "BSE"]:
    else:
        with pd.HDFStore(ifilepath) as hdf_store:
            header_info = hdf_store.get_storer('data').attrs.metadata # type: ignore
            dat = hdf_store.get('data')

    header = pd.DataFrame.from_dict([header_info]) # type: ignore
    return dat, header

# Grab evolutionary states
def select_evolutionary_states(d):
    '''Selects the WDMS and DWD populations at the formation of the first and second white dwarfs

    Parameters
    ----------
    d : `pandas.DataFrame`
        contains T0 data for binaries as specified by BinCodex

    Returns
    -------
    ZAMS : `pandas.DataFrame`
        T0 columns for Zero Age Main Sequence binaries

    WDMS : `pandas.DataFrame`
        T0 columns for WDMS binaries at the formation of the 1st WD

    DWD : `pandas.DataFrame`
        T0 columns for DWD binaries at the formation of the 2nd WD
    '''

    ZAMS = d.groupby('ID', as_index=False).first()

    # the extra 12 handles BPASS
    WDMS1 = d.loc[((d.type1.isin([21,22,23]) & (d.type2.isin([12, 121])))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    WDMS2 = d.loc[((d.type2.isin([21,22,23]) & (d.type1.isin([12, 121])))) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    WDMS = pd.concat([WDMS1, WDMS2])
    DWD = d.loc[(d.type1.isin([21,22,23])) & (d.type2.isin([21,22,23])) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    
    # this handles ComBinE
    if len(DWD) == 0:
        WDMS1 = d.loc[((d.type1 == 2) & (d.type2 == 121)) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
        WDMS2 = d.loc[((d.type2 == 2) & (d.type1 == 121)) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()
    
        WDMS = pd.concat([WDMS1, WDMS2])
        DWD = d.loc[(d.type1 == 2) & (d.type2 == 2) & (d.semiMajor > 0)].groupby('ID', as_index=False).first()

    return ZAMS, WDMS, DWD