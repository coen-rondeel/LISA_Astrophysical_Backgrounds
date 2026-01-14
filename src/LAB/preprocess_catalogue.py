import pandas as pd
import jax.numpy as jnp
from physics import *


class ProprecessCatalogue():
    # Here we should also handle the different metallicity imports
    def __init__(self, config: dict) -> None:
        self.config = config
        self.total_population_mass = float(self.config['population']['total_population_mass'])
        
        self.simple_single_Z_import()
        

        
    def simple_single_Z_import(self) -> None:
        population_df = pd.read_csv(self.config['population']['population_path'])
        population_df = population_df.apply(pd.to_numeric, errors='coerce')  # 'coerce' will turn invalid parsing into NaN

        self.t0 = jnp.array(population_df['t0'].values)

        self.M_ch = chirp_mass(jnp.array(population_df['m1'].values), jnp.array(population_df['m2'].values))

        self.M_ch_pow = jnp.cbrt(self.M_ch ** 5)  

        self.K_factor = K_factor(self.M_ch)

        self.nu0 = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        jnp.array(population_df['a'].values)
                                        )

        self.numax = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        a_min(jnp.array(population_df['m1'].values), 
                                              jnp.array(population_df['m2'].values)
                                              )
                                        )
        
        self.merger_time = tau_GW(2*self.nu0, 2*self.numax, self.K_factor)



    def mock_BH_import(self) -> None:
        population_df = pd.read_csv(self.config['population']['population_path'], usecols=['t0', 'a', 'm1', 'm2'])


        self.t0 = jnp.array(population_df['t0'].values)

        self.M_ch = chirp_mass(jnp.array(population_df['m1'].values), jnp.array(population_df['m2'].values))

        self.M_ch_pow = jnp.cbrt(self.M_ch ** 5)  

        self.K_factor = K_factor(self.M_ch)

        self.nu0 = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        jnp.array(population_df['a'].values)
                                        )

        self.numax = orbital_freq_kepler(jnp.array(population_df['m1'].values), 
                                        jnp.array(population_df['m2'].values), 
                                        a_min_BHs(jnp.array(population_df['m1'].values), 
                                              jnp.array(population_df['m2'].values)
                                              )
                                        )
        
        self.merger_time = tau_GW(2*self.nu0, 2*self.numax, self.K_factor)
        

