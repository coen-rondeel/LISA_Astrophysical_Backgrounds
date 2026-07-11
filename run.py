from lisaastrophysicalbackgrounds.gravitational_wave_background import GravitationalWaveBackground

gwb = GravitationalWaveBackground('base_config.yaml')



gwb.calculate_GWB()

frequencies = gwb.f_vals
omega_gw = gwb.omega_f
num_sources = gwb.N_sources_f