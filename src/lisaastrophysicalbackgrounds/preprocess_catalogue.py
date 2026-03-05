import pandas as pd
from pandas import DataFrame
import jax.numpy as jnp
from .physics import *


class ProprecessCatalogue():
    """The ProprocessCatalogue instance.
    """
    def __init__(self, config: dict) -> None:
        """Initializes the PC class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
        """
        self.config: dict = config
        self.total_population_mass = float(self.config['population']['total_population_mass'])
        
        self.simple_single_Z_import()
        

        
    def simple_single_Z_import(self) -> None:
        """The simplest possible import of a galaxy catalogue of BWDs with only one metallicity.
        """
        population_df = pd.read_csv(self.config['population']['population_path'])
        population_df: DataFrame = population_df.apply(pd.to_numeric, errors='coerce')  # 'coerce' will turn invalid parsing into NaN

        self.t0 = jnp.array(population_df['t0'].values)

        self.M_ch: jax.Array = chirp_mass(jnp.array(population_df['m1'].values), jnp.array(population_df['m2'].values))

        self.M_ch_pow = jnp.cbrt(self.M_ch ** 5)  

        self.K_factor: jax.Array = K_factor(self.M_ch)

        self.nu0: jax.Array = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        jnp.array(population_df['a'].values)
                                        )

        self.numax: jax.Array = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        a_min(jnp.array(population_df['m1'].values), 
                                              jnp.array(population_df['m2'].values)
                                              )
                                        )
        
        self.merger_time: jax.Array = tau_GW(2*self.nu0, 2*self.numax, self.K_factor)



    def mock_BH_import(self) -> None:
        """The simplest possible import of a galaxy catalogue of BBHs with only one metallicity.
        """
        population_df = pd.read_csv(self.config['population']['population_path'], usecols=['t0', 'a', 'm1', 'm2'])

        self.t0 = jnp.array(population_df['t0'].values)

        self.M_ch: jax.Array = chirp_mass(jnp.array(population_df['m1'].values), jnp.array(population_df['m2'].values))

        self.M_ch_pow = jnp.cbrt(self.M_ch ** 5)  

        self.K_factor: jax.Array = K_factor(self.M_ch)

        self.nu0: jax.Array = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        jnp.array(population_df['a'].values)
                                        )

        self.numax: jax.Array = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        a_min_BHs(jnp.array(population_df['m1'].values), 
                                              jnp.array(population_df['m2'].values)
                                              )
                                        )
        
        self.merger_time: jax.Array = tau_GW(2*self.nu0, 2*self.numax, self.K_factor)
        

