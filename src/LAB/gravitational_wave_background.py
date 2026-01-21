import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from background_cosmology import BackgroundCosmology
from preprocess_catalogue import ProprecessCatalogue
from star_formation_history import StarFormationHistory
from physics import *
from utils import *

import h5py
import os
from typing import Tuple, cast
from astropy.cosmology import FLRW

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))



class GravitationalWaveBackground():
    """GWB Class instance
    """
    def __init__(self, config_file: str) -> None:
        """Main class to calculate the gravitational wave background from a given catalogue of ultra-compact binaries.  

        Args:
            config_file (str): Full path to the configuration file.
        """

        os.chdir(MAIN_DIR)

        self.config: dict = get_config(config_file)

        self.get_frequencies()
        self.cosmology = BackgroundCosmology(self.config)
        self.catalogue = ProprecessCatalogue(self.config)
        self.SFH = StarFormationHistory(self.config, self.cosmology.cosmo)

        self.clean_catalogue()



    def get_frequencies(self) -> None:
        """Sets up the frequency relevant quantities according to the configuration file. 
        """

        self.f_min = float(self.config['global']['frequency']['f_min'])
        self.f_max = float(self.config['global']['frequency']['f_max'])
        self.N_fbins = int(float(self.config['global']['frequency']['N_fbins']))

        if self.config['global']['frequency']['f_scale'] == 'linear':
            f_grid = jnp.linspace(self.f_min, self.f_max, 2*self.N_fbins+1)

        elif self.config['global']['frequency']['f_scale'] == 'log10':
            f_grid = jnp.logspace(jnp.log10(self.f_min), jnp.log10(self.f_max), 2*self.N_fbins+1, base=10)

        self.f_grid: jax.Array = f_grid
        self.f_vals: jax.Array = self.f_grid[1::2]
        self.f_bins: jax.Array = self.f_grid[::2]

        self.f_factors: jax.Array = self.f_vals * (self.f_bins[1:]**(2/3) - self.f_bins[:-1]**(2/3)) / (self.f_bins[1:] - self.f_bins[:-1])



    def clean_catalogue(self) -> None:
        """Removes binaries that will never reach the minimum frequency within the age of the considered universe. 
        """

        lookbacktime_func = getattr(self.cosmology.cosmo, "lookback_time")
        max_age: jax.Array = jnp.array(lookbacktime_func(self.cosmology.z_max).value) * 1000 # in Myr
        tau_max: jax.Array = tau_GW(2*self.catalogue.nu0, self.f_min, self.catalogue.K_factor)

        mask: jax.Array = tau_max  < max_age # + self.catalogue.t0 not certain if this should be included, it is not in seppe's code

        self.catalogue.t0 = self.catalogue.t0[mask]
        self.catalogue.M_ch = self.catalogue.M_ch[mask]
        self.catalogue.M_ch_pow = self.catalogue.M_ch_pow[mask]
        self.catalogue.K_factor = self.catalogue.K_factor[mask]
        self.catalogue.nu0 = self.catalogue.nu0[mask]
        self.catalogue.numax = self.catalogue.numax[mask]
        self.catalogue.merger_time = self.catalogue.merger_time[mask]

    

    def calculate_GWB(self) -> None:
        """Main function to calculate the gravitational wave background from the catalogue.
        """
  
        self.prefactor_bulk: float = 8.10e-9 / self.catalogue.total_population_mass
        self.prefactor_birth_merger: float = 1.28e-8 / self.catalogue.total_population_mass

        print("Calculating gravitational wave background...")
        self.calculate_births()
        self.calculate_bulk()
        self.calculate_mergers()

        self.combine_contributions()
        print("Calculation complete...")

        if self.config['global']['save_results']:
            self.save_results()
        


    # main part of the code that calculates the GWB
    def calculate_bulk(self) -> None:
        """Calculates the total bulk contribution to the gravitational wave background.
        """

        @jax.jit
        def get_sum_for_one_bin_and_z(
                z: jax.Array, 
                age: jax.Array, 
                t_max_z: jax.Array, 
                f_low_obs: jax.Array, 
                f_high_obs: jax.Array
            ) -> Tuple[jax.Array, jax.Array]:
            """Calculates the partial bulk contribution of the catalogue to a specific redshift and frequency bin.

            Args:
                z (jax.Array): The redshift value of the considered bin.
                age (jax.Array): The age of universe at the considered bin in Myr.
                t_max_z (jax.Array): The relative age of the universe comparing the age of the considered bin to z_max in Myr.
                f_low_obs (jax.Array): The lower frequency of the considered frequency bin in Hz.
                f_high_obs (jax.Array): The upper frequency of the considered frequency bin in Hz.

            Returns:
                Tuple[jax.Array, jax.Array]: The partial bulk contribution of the redshift and frequency bin for: the GWB strength Omega and the number of systems.
            """

            f_low_zbin: jax.Array = f_low_obs * (1 + z)
            f_high_zbin: jax.Array = f_high_obs * (1 + z)

            # mask to focus on systems that contribute to this bin
            mask: jax.Array = (2 * self.catalogue.nu0 <= f_low_zbin) & (2 * self.catalogue.numax >= f_high_zbin)

            tau: jax.Array = tau_GW(2 * self.catalogue.nu0, f_high_zbin, self.catalogue.K_factor)
            time_since_ZAMS: jax.Array = tau + self.catalogue.t0
            
            # mask to only include systems that had time to evolve into the bin (i.e. not older than universe at max_z)
            mask: jax.Array = mask & (time_since_ZAMS <= t_max_z)

            psi: jax.Array = self.SFH.delayed_SFH(age, time_since_ZAMS) 
            
            # contribution to the GWB 
            partial_omega = jnp.sum(jnp.where(mask, psi * self.catalogue.M_ch_pow, 0.0))
            
            tau_bin: jax.Array = tau_GW(f_low_zbin, f_high_zbin, self.catalogue.K_factor)
            partial_num_syst = jnp.sum(jnp.where(mask, psi * tau_bin * 1e6, 0.0))

            return partial_omega, partial_num_syst


        vmap_z = jax.vmap(get_sum_for_one_bin_and_z, in_axes=(0, 0, 0, None, None))
        vmap_fz = jax.vmap(vmap_z, in_axes=(None, None, None, 0, 0))

        partial_omega_fz, partial_num_syst_fz = vmap_fz(
            self.cosmology.z_vals,
            self.cosmology.ages,
            self.cosmology.z_time_since_z_max,
            self.f_bins[:-1],
            self.f_bins[1:]
        )

        self.omega_bulk_fz: jax.Array = self.prefactor_bulk * partial_omega_fz * \
                                        (1 + self.cosmology.z_vals)**(-4/3) * \
                                        self.cosmology.z_widths * \
                                        self.f_factors[:, None]
        
        self.N_sources_bulk_fz: jax.Array = partial_num_syst_fz / self.catalogue.total_population_mass * \
                                            4 * jnp.pi * self.cosmology.DC_vals**2 * \
                                            self.cosmology.z_widths



    def calculate_births(self) -> None:
        """Calculates the total birth contribution to the gravitational wave background.
        """

        @jax.jit
        def calculate_redshift_birth(z: jax.Array, age: jax.Array, t_max_z: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """Calculates the partial birth contribution of the catalogue to a specific redshift and frequency bin.

            Args:
                z (jax.Array): The redshift value of the considered bin.
                age (jax.Array): The age of universe at the considered bin in Myr.
                t_max_z (jax.Array): The relative age of the universe comparing the age of the considered bin to z_max in Myr.

            Returns:
                Tuple[jax.Array, jax.Array]: The partial birth contribution of the redshift and frequency bin for: the GWB strength Omega and the number of systems.
            """

            f_birth_obs: jax.Array = (2 * self.catalogue.nu0) / (1 + z)
            psi: jax.Array = self.SFH.delayed_SFH(age, self.catalogue.t0)

            bin_idx = jnp.searchsorted(self.f_bins, f_birth_obs, side='right') - 1

            # mask for systems that are born within frequency range and are not older than universe at max_z
            within_band: jax.Array = (bin_idx >= 0) & (bin_idx < self.N_fbins)
            age_OK: jax.Array = (self.catalogue.t0 <= t_max_z)
            mask: jax.Array = within_band & age_OK

            frequency_mapping = jnp.where(mask, bin_idx, 0)
            f_low_obs: jax.Array = self.f_bins[frequency_mapping]
            f_high_obs: jax.Array = self.f_bins[frequency_mapping + 1]
            f_center_obs: jax.Array = self.f_vals[frequency_mapping]

            tau_to_upper_edge: jax.Array = tau_GW(2 * self.catalogue.nu0, f_high_obs * (1+z), self.catalogue.K_factor)
            max_evolve_time: jax.Array = t_max_z - self.catalogue.t0

            # mask for systems that can reach the upper frequency edge within the time available
            mask_reach_edge: jax.Array = tau_to_upper_edge >= max_evolve_time
            tau_in_bin = jnp.where(mask_reach_edge, max_evolve_time, tau_to_upper_edge)

            # calculating the reached frequency after evolving for tau_in_bin
            nu_reached_zbin = jnp.where(mask_reach_edge, 
                                    orbital_freq_from_time(self.catalogue.nu0, 
                                                                max_evolve_time, 
                                                                self.catalogue.K_factor),
                                    f_high_obs * (1 + z) * 0.5,
                                    )
            nu_reached_zbin = nu_reached_zbin[0] if isinstance(nu_reached_zbin, tuple) else nu_reached_zbin
            
            # calculate birth frequency factors
            nu_reached_zbin_23 = jnp.square(jnp.cbrt(nu_reached_zbin))
            nu0_23 = jnp.square(jnp.cbrt(self.catalogue.nu0))
            freq_fac: jax.Array = (nu_reached_zbin_23 - nu0_23) / (f_high_obs - f_low_obs)

            # calculate omega and N_sources contributions of each system
            omega_weights: jax.Array = (f_center_obs * self.catalogue.M_ch_pow * freq_fac * (1/(1+z)) * psi)
            omega_weights = jnp.where(mask, omega_weights, 0.0)
            num_syst_weights = jnp.where(mask, psi * tau_in_bin * 1e6, 0.0)

            # accumulate contributions in the right frequency bins
            partial_omega = jnp.zeros(self.N_fbins).at[frequency_mapping].add(omega_weights)
            partial_num_syst = jnp.zeros(self.N_fbins).at[frequency_mapping].add(num_syst_weights)

            return partial_omega, partial_num_syst
        

        vmap_redshift = jax.vmap(calculate_redshift_birth, in_axes=(0, 0, 0))

        partial_omega_zf, partial_num_syst_zf = vmap_redshift(
            self.cosmology.z_vals,
            self.cosmology.ages,
            self.cosmology.z_time_since_z_max
        )

        partial_omega_fz = jnp.transpose(partial_omega_zf)
        partial_num_syst_fz = jnp.transpose(partial_num_syst_zf)

        self.omega_birth_fz: jax.Array = self.prefactor_birth_merger * partial_omega_fz * \
                                        (1 + self.cosmology.z_vals)**(-1) * \
                                        self.cosmology.z_widths
        
        self.N_sources_birth_fz: jax.Array = partial_num_syst_fz / self.catalogue.total_population_mass * \
                                            4 * jnp.pi * self.cosmology.DC_vals**2 * \
                                            self.cosmology.z_widths



    def calculate_mergers(self) -> None:
        """Calculates the total merger contribution to the gravitational wave background.
        """

        @jax.jit
        def calculate_redshift_merger(z: jax.Array, age: jax.Array, t_max_z: jax.Array) -> Tuple[jax.Array, jax.Array]:
            """Calculates the partial merger contribution of the catalogue to a specific redshift and frequency bin.

            Args:
                z (jax.Array): The redshift value of the considered bin.
                age (jax.Array): The age of universe at the considered bin in Myr.
                t_max_z (jax.Array): The relative age of the universe comparing the age of the considered bin to z_max in Myr.

            Returns:
                Tuple[jax.Array, jax.Array]: The partial merger contribution of the redshift and frequency bin for: the GWB strength Omega and the number of systems.
            """

            evolve_time: jax.Array = t_max_z - self.catalogue.t0
            merger_reached: jax.Array = self.catalogue.merger_time <= evolve_time

            f_now_zbin = jnp.where(merger_reached,
                                2 * self.catalogue.numax,
                                2 * orbital_freq_from_time(self.catalogue.nu0,
                                                            evolve_time,
                                                            self.catalogue.K_factor))
            
            f_now_zbin = f_now_zbin[0] if isinstance(f_now_zbin, tuple) else f_now_zbin
            
            f_now_obs: jax.Array = f_now_zbin / (1 + z)

            bin_idx = jnp.searchsorted(self.f_bins, f_now_obs, side='right') - 1

            # mask for systems that merge within frequency range and binary is born
            within_band: jax.Array = (bin_idx >= 0) & (bin_idx < self.N_fbins)
            binary_born: jax.Array = (evolve_time >=0)
            mask: jax.Array = (within_band & binary_born) 

            # Also not considering binaries that merge outside of the considered frequency range
            mask: jax.Array = mask & (f_now_obs <= 2 * self.catalogue.numax / (1+z))

            # make sure that we do not overcount mergers that are included in the birth contribution
            f_low_obs: jax.Array = self.f_bins[jnp.where(mask, bin_idx, 0)]
            f_low_zbin: jax.Array = f_low_obs * (1 + z)
            mask: jax.Array = mask & (2 * self.catalogue.nu0 < f_low_zbin)

            # Physics for the binaries that reached the merger
            tau_merged: jax.Array = self.catalogue.merger_time
            psi_merged: jax.Array = self.SFH.delayed_SFH(age, tau_merged)
            nu_max_23 = jnp.square(jnp.cbrt(self.catalogue.numax))
            nu_low_23 = jnp.square(jnp.cbrt(f_low_zbin * 0.5))

            # Same for the non-merged binaries
            tau_non_merged: jax.Array = tau_GW(2 * self.catalogue.nu0, f_low_zbin, self.catalogue.K_factor)
            psi_non_merged: jax.Array = self.SFH.delayed_SFH(age, tau_non_merged)
            nu_now_zbin_23 = jnp.square(jnp.cbrt(f_now_zbin * 0.5))

            psi = jnp.where(merger_reached, psi_merged, psi_non_merged)

            # frequency factor for merger case:
            nu_upp_val_23 = jnp.where(merger_reached, nu_max_23, nu_now_zbin_23)
            f_high_obs: jax.Array = self.f_bins[jnp.where(mask, bin_idx + 1, 1)]
            freq_fac: jax.Array = (nu_upp_val_23 - nu_low_23) / (f_high_obs - f_low_obs)

            tau_in_bin = jnp.where(
                    merger_reached,
                    tau_GW(f_low_zbin, 2 * self.catalogue.numax, self.catalogue.K_factor),
                    evolve_time - tau_non_merged
                )
            tau_in_bin = tau_in_bin[0] if isinstance(tau_in_bin, tuple) else tau_in_bin

            f_center_obs: jax.Array = self.f_vals[jnp.where(mask, bin_idx, 0)]
        
            # calculate omega and N_sources contributions of each system
            omega_weights: jax.Array = f_center_obs * self.catalogue.M_ch_pow * freq_fac * (1/(1+z)) * psi
            omega_weights = jnp.where(mask, omega_weights, 0.0)
            
            num_syst_weights = jnp.where(mask, psi * tau_in_bin * 1e6, 0.0)

            # accumulate contributions in the right frequency bins
            frequency_mapping = jnp.where(mask, bin_idx, 0)
            partial_omega = jnp.zeros(self.N_fbins).at[frequency_mapping].add(omega_weights)
            partial_num_syst = jnp.zeros(self.N_fbins).at[frequency_mapping].add(num_syst_weights)

            return partial_omega, partial_num_syst
        

        vmap_redshift = jax.vmap(calculate_redshift_merger, in_axes=(0, 0, 0))
    
        partial_omega_zf, partial_num_syst_zf = vmap_redshift(
            self.cosmology.z_vals,
            self.cosmology.ages,
            self.cosmology.z_time_since_z_max
        )

        partial_omega_fz = jnp.transpose(partial_omega_zf)
        partial_num_syst_fz = jnp.transpose(partial_num_syst_zf)

        self.omega_merger_fz: jax.Array = self.prefactor_birth_merger * partial_omega_fz * \
                                        (1 + self.cosmology.z_vals)**(-1) * \
                                        self.cosmology.z_widths
        
        self.N_sources_merger_fz: jax.Array = partial_num_syst_fz / self.catalogue.total_population_mass * \
                                            4 * jnp.pi * self.cosmology.DC_vals**2 * \
                                            self.cosmology.z_widths


    def combine_contributions(self) -> None:
        """Combines the different contributions into one gravitational wave background.
        """
        self.omega_fz: jax.Array = self.omega_bulk_fz + self.omega_birth_fz + self.omega_merger_fz
        self.N_sources_fz: jax.Array = self.N_sources_bulk_fz + self.N_sources_birth_fz + self.N_sources_merger_fz

        self.omega_f = jnp.sum(self.omega_fz, axis=1)
        self.N_sources_f = jnp.sum(self.N_sources_fz, axis=1)


    def save_results(self) -> None:
        """Saves the results of the gravitational wave background calculation to an HDF5 file and generates a plot.
        """
        print("Saving results at " + self.config['global']['save_directory'])
        save_directory: str = self.config['global']['save_directory']
        data_filename: str = f'{self.config['population']['catalogue_name']}_gwb_results.h5'

        with h5py.File(save_directory + data_filename, 'w') as hf:
            frequency_data = hf.create_dataset('f_vals', data=self.f_vals)
            frequency_data.attrs['f_min'] = self.f_min
            frequency_data.attrs['f_max'] = self.f_max
            frequency_data.attrs['N_fbins'] = self.N_fbins
            frequency_data.attrs['frequency_scale'] = self.config['global']['frequency']['f_scale']

            redshift_data = hf.create_dataset('z_vals', data=self.cosmology.z_vals)
            redshift_data.attrs['z_min'] = self.cosmology.z_min
            redshift_data.attrs['z_max'] = self.cosmology.z_max
            redshift_data.attrs['N_zbins'] = self.cosmology.N_zbins
            redshift_data.attrs['z_scale'] = self.cosmology.z_scale
            redshift_data.attrs['cosmology'] = str(self.cosmology.cosmo)

            hf.attrs['catalogue'] = self.config['population']['catalogue_name']
            hf.create_dataset('omega_fz', data=self.omega_fz)
            hf.create_dataset('N_sources_fz', data=self.N_sources_fz)
            hf.create_dataset('omega_f', data=self.omega_f)
            hf.create_dataset('N_sources_f', data=self.N_sources_f)

        plot_filename: str = f'GWB_for_{self.config['population']['catalogue_name']}_with_{self.config['SFR']['SFR_name']}.png'
        get_GWB_plot(self.f_vals, self.omega_f, save_path=save_directory + plot_filename)



