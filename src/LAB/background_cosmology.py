import astropy.cosmology as cosmo_module
import jax.numpy as jnp


class BackgroundCosmology():

    def __init__(self, config: dict) -> None:
        self.config = config

        self.get_cosmology()
        self.get_redshift_grid()

    def get_cosmology(self) -> None:

        if self.config['cosmology']['standard']:

            cosmo_name = self.config['cosmology']['standard_cosmology'] 
            
            with cosmo_module.default_cosmology.set(cosmo_name):
                cosmo = cosmo_module.default_cosmology.get()

            self.cosmo = cosmo
        
        else:

            cosmo_name =  self.config['cosmology']['custom_cosmology']['cosmology_type']

            CosmoClass = getattr(cosmo_module, cosmo_name)

            cosmo = CosmoClass(**self.config['cosmology']['custom_cosmology']['params'])

            self.cosmo = cosmo
        
    
    def get_redshift_grid(self) -> None:
        self.z_min = float(self.config['cosmology']['z_min'])
        self.z_max = float(self.config['cosmology']['z_max'])
        self.N_zbins = int(float(self.config['cosmology']['N_zbins']))
        self.z_scale = self.config['cosmology']['z_scale']

        if self.z_scale == 'linear':
            z_grid = jnp.linspace(self.z_min, self.z_max, 2*self.N_zbins + 1)
                
        elif self.z_scale == 'log10':
            z_grid = jnp.logspace(jnp.log10(self.z_min), jnp.log10(self.z_max), 2*self.N_zbins + 1, base=10)

        elif self.z_scale == 'log':
            z_grid = jnp.logspace(jnp.log(self.z_min), jnp.log(self.z_max), 2*self.N_zbins + 1, base=jnp.e)

        elif self.z_scale == 'power':
            self.z_power_scale = self.config['cosmology']['z_power_scale']

            if not(jnp.isnan(self.z_power_scale)):
                loga_factor = jnp.log(self.z_power_scale)

                z_grid = jnp.logspace(jnp.log(self.z_min) / loga_factor, 
                                    jnp.log(self.z_max) / loga_factor,
                                    2*self.N_zbins + 1, 
                                    base=self.z_power_scale)
            
            else: 
                raise TypeError(f"Unsupported z_power_scale must be a number")
            
        else: 
            raise ValueError(f"Unsupported z_scale type: {self.z_scale}")


        self.z_grid = z_grid
        self.z_vals = z_grid[1::2]
        self.z_bins = z_grid[::2]

        self.DC_vals = self.cosmo.comoving_distance(self.z_vals).value

        self.z_widths = jnp.diff(jnp.array(self.cosmo.comoving_distance(self.z_bins).value))

        lookback_times = jnp.array(self.cosmo.lookback_time(self.z_vals).value * 1000)

        self.z_time_since_z_max = self.cosmo.lookback_time(self.z_max).value * 1000 - lookback_times

        self.ages = self.cosmo.age(0).value * 1000 - lookback_times

    



