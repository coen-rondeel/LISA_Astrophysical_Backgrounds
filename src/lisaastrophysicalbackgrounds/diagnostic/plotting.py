"""Plotting functions for visualizing binary populations and GWB calculation results."""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Premium plot styling defaults
PLOTTING_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 9,
    "figure.titlesize": 13,
}


def _apply_style():
    """Apply the premium plot styling configurations."""
    plt.rcParams.update(PLOTTING_STYLE)


def plot_population_properties(population, save_path=None) -> plt.Figure:
    """Plot physical properties of the input binary population.

    Generates distributions of chirp mass, initial orbital frequency,
    and a mass scatter plot if components are available.

    Args:
        population: The PreprocessPopulation instance containing the population data.
        save_path: Optional path or filename where the plot will be saved.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    _apply_style()

    # Extract properties safely, converting JAX arrays to NumPy arrays
    t0 = np.array(population.t0) if population.t0 is not None else None
    m1 = np.array(population.m1) if population.m1 is not None else None
    m2 = np.array(population.m2) if population.m2 is not None else None
    m_ch = np.array(population.M_ch) if getattr(population, "M_ch", None) is not None else None
    nu0 = np.array(population.nu0) if getattr(population, "nu0", None) is not None else None
    tau = np.array(population.merger_time) if getattr(population, "merger_time", None) is not None else None
    Z = np.array(population.Z) if population.Z is not None else None

    has_masses = m1 is not None and m2 is not None
    num_subplots = 3 if has_masses else 2
    
    fig, axes = plt.subplots(1, num_subplots, figsize=(4 * num_subplots, 3.5), constrained_layout=True)
    if num_subplots == 1:
        axes = [axes]

    color_palette = ["#0284c7", "#f59e0b", "#10b981", "#8b5cf6"]

    # Plot 1: Chirp Mass Distribution
    ax_idx = 0
    if m_ch is not None and len(m_ch) > 0:
        axes[ax_idx].hist(m_ch, bins=30, color=color_palette[0], edgecolor="white", alpha=0.8)
        axes[ax_idx].set_xlabel(r"Chirp Mass $M_{\rm ch}$ [$M_\odot$]")
        axes[ax_idx].set_ylabel("Count")
        axes[ax_idx].set_title("Chirp Mass Distribution")
        axes[ax_idx].grid(True)
        ax_idx += 1

    # Plot 2: Frequency/Period Distribution
    if nu0 is not None and len(nu0) > 0:
        f_gw = 2 * nu0  # GW frequency is twice orbital frequency
        axes[ax_idx].hist(f_gw, bins=np.logspace(np.log10(max(1e-6, f_gw.min())), np.log10(f_gw.max()), 30),
                          color=color_palette[1], edgecolor="white", alpha=0.8)
        axes[ax_idx].set_xscale("log")
        axes[ax_idx].set_xlabel(r"Initial GW Frequency $f_{\rm gw,0}$ [Hz]")
        axes[ax_idx].set_ylabel("Count")
        axes[ax_idx].set_title("Initial Frequency Distribution")
        axes[ax_idx].grid(True, which="both")
        ax_idx += 1

    # Plot 3: Mass 1 vs Mass 2 Scatter
    if has_masses and m1 is not None and m2 is not None and len(m1) > 0:
        # Use scatter or hexbin depending on size
        if len(m1) > 5000:
            hb = axes[ax_idx].hexbin(m1, m2, gridsize=25, cmap="Blues", mincnt=1)
            fig.colorbar(hb, ax=axes[ax_idx], label="Count")
        else:
            axes[ax_idx].scatter(m1, m2, color=color_palette[2], alpha=0.5, edgecolors="none", s=15)
        axes[ax_idx].set_xlabel(r"Primary Mass $m_1$ [$M_\odot$]")
        axes[ax_idx].set_ylabel(r"Secondary Mass $m_2$ [$M_\odot$]")
        axes[ax_idx].set_title(r"$m_1$ vs $m_2$ Distribution")
        axes[ax_idx].grid(True)

    fig.suptitle(f"Binary Population Diagnostics: {population._config['population']['population_name']}", y=1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved population properties plot to {save_path}")

    return fig


def plot_gwb_spectral_components(gwb, save_path=None) -> plt.Figure:
    """Plot the GWB spectrum, showing the total and separated physical components.

    Separates the background into Bulk, Birth, and Mergers, including uncertainty
    intervals if variance is available.

    Args:
        gwb: The GravitationalWaveBackground instance after running calculate_GWB().
        save_path: Optional path or filename where the plot will be saved.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    _apply_style()

    frequencies = np.array(gwb.f_vals)
    omega_f = np.array(gwb.omega_f)
    var_f = np.array(gwb.var_f) if getattr(gwb, "var_f", None) is not None else None

    # Compute component lines
    omega_bulk_f = np.array(np.sum(gwb.omega_bulk_fzZ, axis=(1, 2)))
    omega_birth_f = np.array(np.sum(gwb.omega_birth_fzZ, axis=(1, 2)))
    omega_merger_f = np.array(np.sum(gwb.omega_merger_fzZ, axis=(1, 2)))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    # Color Palette: Premium/Clean
    c_total = "#0f172a"  # Slate 900 (Total)
    c_bulk = "#3b82f6"   # Blue 500 (Bulk)
    c_birth = "#f59e0b"  # Amber 500 (Birth)
    c_merger = "#10b981" # Emerald 500 (Merger)

    # Plot uncertainty bands first (background layers)
    if var_f is not None and np.any(var_f > 0):
        std_f = np.sqrt(var_f)
        # Prevent negative values in log scale
        ax.fill_between(frequencies, np.maximum(1e-20, omega_f - 3.0 * std_f), omega_f + 3.0 * std_f,
                        color="#38bdf8", alpha=0.1, label=r"$3\sigma$ Interval")
        ax.fill_between(frequencies, np.maximum(1e-20, omega_f - 2.0 * std_f), omega_f + 2.0 * std_f,
                        color="#38bdf8", alpha=0.15, label=r"$2\sigma$ Interval")
        ax.fill_between(frequencies, np.maximum(1e-20, omega_f - std_f), omega_f + std_f,
                        color="#38bdf8", alpha=0.25, label=r"$1\sigma$ Interval")

    # Plot individual components
    ax.loglog(frequencies, omega_bulk_f, label="Bulk", color=c_bulk, linestyle="--", linewidth=1.5)
    ax.loglog(frequencies, omega_birth_f, label="Birth", color=c_birth, linestyle=":", linewidth=1.5)
    ax.loglog(frequencies, omega_merger_f, label="Mergers", color=c_merger, linestyle="-.", linewidth=1.5)

    # Plot total background
    ax.loglog(frequencies, omega_f, label="Total GWB", color=c_total, linestyle="-", linewidth=2.0)

    ax.set_xlabel(r"Frequency $f$ [Hz]")
    ax.set_ylabel(r"GWB strength $\Omega_{\rm GW}(f)$")
    ax.set_title(f"GWB Spectral Contributions: {gwb.config['population']['population_name']}")
    ax.grid(True, which="both", alpha=0.3)
    
    # Legend settings
    ax.legend(loc="lower left", framealpha=0.9)
    
    # Sensible axis bounds
    ax.set_xlim(left=float(frequencies.min()), right=float(frequencies.max()))
    # Set y limits based on data range, fallback if needed
    non_zero_omega = omega_f[omega_f > 0]
    if len(non_zero_omega) > 0:
        ymin = max(1e-15, 10 ** (np.floor(np.log10(non_zero_omega.min())) - 1))
        ymax = 10 ** (np.ceil(np.log10(omega_f.max())) + 1)
        ax.set_ylim(bottom=ymin, top=ymax)
    else:
        ax.set_ylim(bottom=1e-15, top=1e-8)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved GWB spectrum plot to {save_path}")

    return fig


def plot_gwb_redshift_evolution(gwb, save_path=None) -> plt.Figure:
    r"""Plot GWB evolution as a function of redshift.

    Generates a 2D intensity map of GWB strength $\Omega_{\rm GW}(f, z)$ and
    a 1D plot of the integrated energy density $\Omega_{\rm GW}(z)$ showing the peak epoch.

    Args:
        gwb: The GravitationalWaveBackground instance.
        save_path: Optional path or filename where the plot will be saved.

    Returns:
        plt.Figure: The matplotlib figure object.
    """
    _apply_style()

    frequencies = np.array(gwb.f_vals)
    z_vals = np.array(gwb.cosmology.z_vals)
    omega_fz = np.array(gwb.omega_fz)  # Shape: (N_fbins, N_zbins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)

    # Plot 1: 2D Heatmap (Redshift vs Frequency)
    # Pivot matrix for pcolormesh: rows=frequency, cols=redshift
    Z_mesh, F_mesh = np.meshgrid(z_vals, frequencies)
    
    # Mask zero/negative elements to avoid log issues in LogNorm
    omega_fz_safe = np.maximum(1e-25, omega_fz)
    
    pcm = axes[0].pcolormesh(
        Z_mesh, F_mesh, omega_fz_safe,
        norm=mcolors.LogNorm(vmin=max(1e-18, omega_fz_safe.max() * 1e-6), vmax=omega_fz_safe.max()),
        cmap="magma",
        shading="auto"
    )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Redshift $z$")
    axes[0].set_ylabel("Frequency $f$ [Hz]")
    axes[0].set_title(r"Differential GWB Strength $\Omega_{\rm GW}(f, z)$")
    fig.colorbar(pcm, ax=axes[0], label=r"$\Omega_{\rm GW}(f, z)$")

    # Plot 2: Integrated GWB vs Redshift (Peak star formation environment visualization)
    # Sum over frequencies to show integrated contribution per redshift bin
    omega_z = np.sum(omega_fz, axis=0)
    axes[1].plot(z_vals, omega_z, color="#4f46e5", linewidth=2.0, marker="o", markersize=4)
    axes[1].set_xlabel("Redshift $z$")
    axes[1].set_ylabel(r"Integrated GWB $\sum_f \Omega_{\rm GW}(f, z)$")
    axes[1].set_title("GWB Contribution vs Redshift")
    axes[1].grid(True)
    if omega_z.max() > 0:
        axes[1].set_ylim(bottom=0, top=omega_z.max() * 1.15)

    fig.suptitle(f"GWB Redshift Evolution: {gwb.config['population']['population_name']}", y=1.05)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved GWB redshift evolution plot to {save_path}")

    return fig


def plot_gwb_metallicity_breakdown(gwb, save_path=None) -> plt.Figure | None:
    """Plot the GWB spectrum broken down by different metallicity contributions.

    Only relevant if the calculation configured multiple metallicities.

    Args:
        gwb: The GravitationalWaveBackground instance.
        save_path: Optional path or filename where the plot will be saved.

    Returns:
        plt.Figure or None: The matplotlib figure object, or None if only one metallicity is present.
    """
    _apply_style()

    frequencies = np.array(gwb.f_vals)
    unique_Zs = np.array(gwb.unique_Zs)
    
    if len(unique_Zs) <= 1:
        print("Skipping metallicity breakdown plot: Single metallicity population.")
        return None

    omega_fZ = np.array(gwb.omega_fZ)  # Shape: (N_fbins, N_Zbins)

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    # Use a distinct colormap for metallicity levels
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 0.85, len(unique_Zs))]

    for idx, Z_val in enumerate(unique_Zs):
        label_text = f"$Z = {Z_val}$" if Z_val >= 0.0001 else f"$Z = {Z_val:.1e}$"
        ax.loglog(frequencies, omega_fZ[:, idx], label=label_text, color=colors[idx], linewidth=1.5)

    # Plot the total for reference
    ax.loglog(frequencies, np.array(gwb.omega_f), label="Total GWB", color="black", linestyle="--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel(r"Frequency $f$ [Hz]")
    ax.set_ylabel(r"GWB strength $\Omega_{\rm GW}(f)$")
    ax.set_title(f"GWB Breakdown by Metallicity: {gwb.config['population']['population_name']}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.9)
    ax.set_xlim(left=float(frequencies.min()), right=float(frequencies.max()))

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved GWB metallicity breakdown plot to {save_path}")

    return fig


def plot_sfrd_vs_redshift(gwb, save_path=None) -> plt.Figure | None:
    """Plot Star Formation Rate Density (SFRD) vs redshift.

    Replicates Figure 1 from Hofman & Nelemans (2024). Shows the SFRD
    as a function of redshift, split by metallicity bins.

    Args:
        gwb: The GravitationalWaveBackground instance.
        save_path: Optional path or filename where the plot will be saved.

    Returns:
        plt.Figure or None: The matplotlib figure object, or None if SFH is missing.
    """
    _apply_style()

    if getattr(gwb, "SFH", None) is None:
        print("Skipping SFRD plot: SFH object is not initialized.")
        return None

    # Generate a redshift grid
    redshifts = np.linspace(0.0, 8.0, 200)
    
    # Get metallicities
    SFH_Zs = np.atleast_1d(np.array(gwb.SFH.SFH_Zs))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)

    # Use a warm color palette like the paper's original plot
    cmap = plt.get_cmap("YlOrRd")
    if len(SFH_Zs) > 1:
        colors = [cmap(i) for i in np.linspace(0.25, 0.95, len(SFH_Zs))]
    else:
        colors = ["firebrick"]

    import jax.numpy as jnp
    z_jax = jnp.array(redshifts)

    for idx, Z_val in enumerate(SFH_Zs):
        try:
            # The SFH functions in StarFormationHistory take (redshifts, metallicity)
            # strolger takes (ages, redshifts, metallicity)
            if gwb.config["SFH"]["SFH_name"] == "strolger":
                # Convert redshifts to age in Myr
                age_func = getattr(gwb.cosmology.cosmo, "age")
                ages = np.array(age_func(redshifts).value * 1000)
                sfrd = gwb.SFH._psi_function(jnp.array(ages), z_jax, Z_val)
            else:
                sfrd = gwb.SFH._psi_function(z_jax, Z_val)
            
            sfrd_np = np.array(sfrd)
            label_text = f"$Z = {Z_val}$" if Z_val >= 0.0001 else f"$Z = {Z_val:.1e}$"
            ax.plot(redshifts, sfrd_np, label=label_text, color=colors[idx], linewidth=2.0)
        except Exception as e:
            print(f"Failed to evaluate SFRD for Z={Z_val}: {e}")

    ax.set_xlabel(r"Redshift ($z$)")
    ax.set_ylabel(r"SFRD ($M_\odot \, \rm{Mpc}^{-3} \, \rm{yr}^{-1}$)")
    
    sfh_name_clean = gwb.config["SFH"]["SFH_name"].replace("_", " ").title()
    ax.set_title(f"Star Formation Rate Density ({sfh_name_clean})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_xlim(0, 8)
    ax.set_ylim(bottom=-0.002)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved SFRD vs Redshift plot to {save_path}")

    return fig


def generate_diagnostic_plots(gwb, save_directory=None) -> dict[str, str]:
    """Generate all diagnostic plots for the simulation and save them.

    Args:
        gwb: The GravitationalWaveBackground instance.
        save_directory: Directory where plots should be saved. If None, uses GWB config path.

    Returns:
        dict[str, str]: Dictionary mapping plot names to their absolute file paths.
    """
    if save_directory is None:
        save_directory = Path(gwb.config["global"]["save_directory"])
    else:
        save_directory = Path(save_directory)

    save_directory.mkdir(parents=True, exist_ok=True)
    pop_name = gwb.config["population"]["population_name"]
    sfh_name = gwb.config["SFH"]["SFH_name"]

    generated_plots = {}

    # 1. Population plots
    pop_plot_path = save_directory / f"diagnostic_population_{pop_name}.png"
    try:
        plot_population_properties(gwb.population, save_path=pop_plot_path)
        generated_plots["population"] = str(pop_plot_path.resolve())
    except Exception as e:
        print(f"Failed to generate population diagnostic plot: {e}")

    # 2. GWB Spectral components
    spec_plot_path = save_directory / f"diagnostic_GWB_spectrum_{pop_name}_with_{sfh_name}.png"
    try:
        plot_gwb_spectral_components(gwb, save_path=spec_plot_path)
        generated_plots["spectral_components"] = str(spec_plot_path.resolve())
    except Exception as e:
        print(f"Failed to generate GWB spectrum diagnostic plot: {e}")

    # 3. GWB Redshift evolution
    redshift_plot_path = save_directory / f"diagnostic_GWB_redshift_{pop_name}.png"
    try:
        plot_gwb_redshift_evolution(gwb, save_path=redshift_plot_path)
        generated_plots["redshift_evolution"] = str(redshift_plot_path.resolve())
    except Exception as e:
        print(f"Failed to generate GWB redshift diagnostic plot: {e}")

    # 4. GWB Metallicity breakdown (if applicable)
    if len(gwb.unique_Zs) > 1:
        metallicity_plot_path = save_directory / f"diagnostic_GWB_metallicity_{pop_name}.png"
        try:
            plot_gwb_metallicity_breakdown(gwb, save_path=metallicity_plot_path)
            generated_plots["metallicity_breakdown"] = str(metallicity_plot_path.resolve())
        except Exception as e:
            print(f"Failed to generate GWB metallicity breakdown plot: {e}")

    # 5. SFRD vs Redshift plot
    if getattr(gwb, "SFH", None) is not None:
        sfrd_plot_path = save_directory / f"diagnostic_SFRD_{sfh_name}.png"
        try:
            plot_sfrd_vs_redshift(gwb, save_path=sfrd_plot_path)
            generated_plots["sfrd_vs_redshift"] = str(sfrd_plot_path.resolve())
        except Exception as e:
            print(f"Failed to generate SFRD diagnostic plot: {e}")

    return generated_plots
