import jax
import jax.numpy as jnp
from typing import Callable, Optional
from astropy.cosmology import Cosmology


class StarFormationHistory():
    """SFH class instance.
    """
    def __init__(self, config: dict, cosmo: Cosmology) -> None:
        """Initializes the SFH class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
            cosmo (Cosmology): The considered cosmology as initialized in the BackgroundCosmolgy class.
        """
        self._config: dict = config

        self.z_high_res = jnp.linspace(0, 12, int(1e4))
        
        age_func = getattr(cosmo, 'age')
        self.age_high_res = jnp.array(age_func(self.z_high_res).value * 1000)  # in Myr
        
        self._flip_age = jnp.flip(self.age_high_res)
        self._flip_z = jnp.flip(self.z_high_res)

        sfh_name = self._config['SFH']['SFH_name']
        self._psi_function: Callable = getattr(self, sfh_name)
        
        if sfh_name == "data":
            self._setup_grid_data()
            

    def delayed_SFH(self, age: jax.Array, delay: jax.Array, metallicity: Optional[jax.Array] = None) -> jax.Array:
        """Calculates the delayed star formation rate for a given age of the universe and the delay of the formed stars.

        Args:
            age (jax.Array): The age of the universe in Myr
            delay (jax.Array): The delay of the considered star since formation. I.e., the difference between the formation time of the stars and time of creation of the GW binary in Myr.
            metallicity (jax.Array): The metallicity of the binary

        Returns:
            jax.Array: The delayed star formation history in Msol / yr / Mpc^3
        """ 
        age_at_star_formation: jax.Array = age - delay # in Myr
        
        z_at_star_formation = jnp.interp(age_at_star_formation, self._flip_age, self._flip_z)

        return self._psi_function(z_at_star_formation, metallicity)


    def madau_and_dickinson(self, redshifts: jax.Array, metallicity: Optional[jax.Array]) -> jax.Array:
        """The madau and dickinson SFR ref: https://doi.org/10.1146/annurev-astro-081811-125615

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The madau and dickinson SFR in Msol / yr / Mpc^3
        """

        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.015 * jnp.power(1 + z, 2.7) / (1 + jnp.power((1 + z) / 2.9, 5.6))
        
        sfr_z = SFH_function(redshifts)
        
        if metallicity is None:
            return sfr_z
        
        p_Z = self.analytical_metallicity_distribution(redshifts, metallicity)

        return sfr_z * p_Z


    def analytical_metallicity_distribution(self, z: jax.Array, metallicity: jax.Array) -> jax.Array:
        """A standard log-normal analytical metallicity distribution P(Z | z).
        (You can update mean_logZ and sigma_logZ to match Langer & Norman 2006 or similar).
        """
        mean_logZ: jax.Array = -0.2 * z 
        sigma_logZ: float = 0.5
        
        logZ = jnp.log10(jnp.where(metallicity > 0, metallicity, 1e-12))
        
        pdf: jax.Array = (1.0 / (sigma_logZ * jnp.sqrt(2 * jnp.pi)) * 
                          jnp.exp(-0.5 * jnp.square((logZ - mean_logZ) / sigma_logZ)))
        return pdf
    
    
    def neijssel_2019(self, z: jax.Array, metallicity: Optional[jax.Array]) -> jax.Array:
        """The Star Formation Rate Density and Metallicity distribution from Neijssel et al. 2019.
        Ref: https://arxiv.org/abs/1906.08136 (Equations 6, 7, and 8)

        Args:
            z (jax.Array): The considered redshifts.
            metallicity (Optional[jax.Array]): The metallicity of the binary.

        Returns:
            jax.Array: The evaluated SFRD in Msol / yr / Mpc^3 (per unit Z if metallicity is provided).
        """
        # (Eq. 6 from Neijssel et al. 2019)
        sfr_z = 0.01 * jnp.power(1.0 + z, 2.77) / (1.0 + jnp.power((1.0 + z) / 2.9, 4.7))
        
        if metallicity is None:
            return sfr_z
            
        # (Eq. 8 from Neijssel et al. 2019)
        mean_Z = 0.035 * jnp.power(10.0, -0.23 * z)
        
        sigma = 0.39
        mu = jnp.log(mean_Z) - (sigma**2) / 2.0
        
        # Safely handle Z <= 0 to prevent NaNs in jnp.log
        Z_safe = jnp.where(metallicity > 0, metallicity, 1e-12)
        ln_Z = jnp.log(Z_safe)
        
        # (Eq. 7 from Neijssel et al. 2019)
        p_Z = (1.0 / (Z_safe * sigma * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
            -0.5 * jnp.square((ln_Z - mu) / sigma)
        )
        # TODO add deltaZ factor
        print("WARNING: deltaZ factor not yet implemented")
        return sfr_z * p_Z
    

    def _setup_grid_data(self) -> None:
        path_to_sfh_data = self._config['SFH']['SFH_path']
