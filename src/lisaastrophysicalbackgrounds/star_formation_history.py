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
        self.get_metallicities()
        
        #* Setup high resolution grid for interpolation
        self.z_high_res = jnp.linspace(0, 10, int(1e4))
        age_func = getattr(cosmo, 'age')
        self.age_high_res = jnp.array(age_func(self.z_high_res).value * 1000)  # in Myr
        self._flip_age = jnp.flip(self.age_high_res)
        self._flip_z = jnp.flip(self.z_high_res)

        #* Get SFH function
        sfh_name = self._config['SFH']['SFH_name']
        
        if sfh_name == "chruslinska_and_nelemans":
            self._setup_chruslinska_data()
        
        self._psi_function: Callable = getattr(self, sfh_name)
        

    
    def get_metallicities(self) -> None:
        """Sets up the metallicity grid using the config file is N_Zs > 1.
        """
        self.SFH_Zs = jnp.atleast_1d(jnp.array(self._config["SFH"]["SFH_metallicities"]))
        
        if len(self.SFH_Zs) > 1:
            self.SFH_Z_bins = jnp.atleast_1d(jnp.array(self._config["SFH"]["SFH_metallicity_bins"]))
            assert len(self.SFH_Z_bins) == len(self.SFH_Zs) + 1, "Metallicity grid not initiallized correctly, please check."
        else: 
            self.SFH_Z_bins = None
        

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

        if self._config['SFH']['SFH_name'] == 'strolger':
            return self._psi_function(age_at_star_formation, z_at_star_formation, metallicity)

        return self._psi_function(z_at_star_formation, metallicity)


    def madau_and_dickinson(self, redshifts: jax.Array, metallicity: Optional[jax.Array]) -> jax.Array:
        """The Madau and Dickinson SFR ref: https://doi.org/10.1146/annurev-astro-081811-125615

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Madau and Dickinson SFR in Msol / yr / Mpc^3
        """

        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.015 * jnp.power(1 + z, 2.7) / (1 + jnp.power((1 + z) / 2.9, 5.6))
        
        sfr_z = SFH_function(redshifts)
        
        if metallicity is None:
            return sfr_z
        
        p_Z = self.analytical_metallicity_distribution(redshifts, metallicity)

        return sfr_z * p_Z


    def madau_and_fragos(self, redshifts: jax.Array, metallicity: Optional[jax.Array]) -> jax.Array:
        """The Madau and Fragos SFR ref: https://doi.org/10.3847/1538-4357/aa6af9

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Madau and Fragos SFR in Msol / yr / Mpc^3
        """

        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.01 * jnp.power(1 + z, 2.6) / (1 + jnp.power((1 + z) / 3.2, 6.2))

        sfr_z = SFH_function(redshifts)

        if metallicity is None:
            return sfr_z

        p_Z = self.analytical_metallicity_distribution(redshifts, metallicity)

        return sfr_z * p_Z


    def strolger(self, ages: jax.Array, redshifts: jax.Array, metallicity: Optional[jax.Array]) -> jax.Array:
        """The Strolger et al. 2004 SFR ref: https://doi.org/10.1086/422901

        Args:
            ages (jax.Array): The considered ages of the Universe
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The Madau and Fragos SFR in Msol / yr / Mpc^3
        """

        def SFH_function(t: jax.Array) -> jax.Array:
            return  0.182 * (jnp.power(t, 1.26) * jnp.exp(-t/1.865) + 0.071 * jnp.exp(0.071 * (t-13.47) /1.865))

        sfr_t = SFH_function(ages *1e-3)

        if metallicity is None:
            return sfr_t

        p_Z = self.analytical_metallicity_distribution(redshifts, metallicity)

        return sfr_t * p_Z


    def analytical_metallicity_distribution(self, z: jax.Array, metallicity: jax.Array) -> jax.Array:
        """A standard log-normal analytical metallicity distribution P(Z | z).
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
        ln_Z = jnp.log(metallicity)
        
        # (Eq. 7 from Neijssel et al. 2019)
        p_Z = (1.0 / (metallicity * sigma * jnp.sqrt(2.0 * jnp.pi))) * jnp.exp(
            -0.5 * jnp.square((ln_Z - mu) / sigma)
        )
        # TODO add deltaZ factor
        print("WARNING: deltaZ factor not yet implemented")
        return sfr_z * p_Z
    
    

    def _setup_chruslinska_data(self) -> None:
        path_to_sfh_data = self._config['SFH']['SFH_path']
        
        
