import jax
import jax.numpy as jnp


class StarFormationHistory():

    def __init__(self, config: dict, cosmo) -> None:
        self.config = config

        self.z_high_res = jnp.linspace(0, 12, int(1e4))
        self.age_high_res = jnp.array(cosmo.age(self.z_high_res).value * 1000)  # in Myr


    def delayed_SFH(self, age: jax.Array, delay: jax.Array) -> jax.Array:

        psi_at_z = getattr(self, self.config['SFR']['SFR_name'])
 
        age_at_star_formation = age - delay # in Myr
        
        z_at_star_formation = jnp.flip(jnp.interp(jnp.flip(age_at_star_formation),
                                                  jnp.flip(self.age_high_res),
                                                  jnp.flip(self.z_high_res),
                                                  ))

        return psi_at_z(z_at_star_formation)


    def madau_and_dickinson(self, redshifts: jax.Array) -> jax.Array:

        @jax.jit
        def SFH_function(z: jax.Array) -> jax.Array:
            return 0.015 * jnp.power(1 + z, 2.7) / (1 + jnp.power((1 + z) / 2.9, 5.6))
        
        return SFH_function(redshifts)



