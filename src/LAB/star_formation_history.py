import jax
import jax.numpy as jnp
from astropy.cosmology import Cosmology


class StarFormationHistory():
    """SFH class instance.
    """
    def __init__(self, config: dict, cosmo: Cosmology) -> None:
        """Initializes the SFR class according to the user defined config file.

        Args:
            config (dict): The yaml imported configuration file as a dictionary.
            cosmo (Cosmology): The considered cosmology as initialized in the BackgroundCosmolgy class.
        """
        self._config: dict = config

        self.z_high_res = jnp.linspace(0, 12, int(1e4))
        self.age_high_res = jnp.array(cosmo.age(self.z_high_res).value * 1000)  # in Myr


    def delayed_SFH(self, age: jax.Array, delay: jax.Array) -> jax.Array:
        """Calculates the delayed star formation rate for a given age of the universe and the delay of the formed stars.

        Args:
            age (jax.Array): The age of the universe in Myr
            delay (jax.Array): The delay of the considered star since formation. I.e., the difference between the formation time of the stars and time of creation of the GW binary in Myr.

        Returns:
            jax.Array: The delayed star formation history in Msol / yr / Mpc^3
        """

        psi_at_z: function = getattr(self, self._config['SFR']['SFR_name'])
 
        age_at_star_formation: jax.Array = age - delay # in Myr
        
        z_at_star_formation = jnp.flip(jnp.interp(jnp.flip(age_at_star_formation),
                                                  jnp.flip(self.age_high_res),
                                                  jnp.flip(self.z_high_res),
                                                  ))

        return psi_at_z(z_at_star_formation)


    def madau_and_dickinson(self, redshifts: jax.Array) -> jax.Array:
        """The madau and dickinson SFR ref: https://doi.org/10.1146/annurev-astro-081811-125615

        Args:
            redshifts (jax.Array): The considered redshifts

        Returns:
            jax.Array: The madau and dickinson SFR in Msol / yr / Mpc^3
        """

        @jax.jit
        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.015 * jnp.power(1 + z, 2.7) / (1 + jnp.power((1 + z) / 2.9, 5.6))
        
        return SFH_function(redshifts)



