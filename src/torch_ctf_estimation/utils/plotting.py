import torch
from ..estimate_defocus_1d import Defocus1DResults
from torch_fourier_filter.ctf import calculate_total_phase_shift
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    print("For plotting please install [plot] extras")
    raise ModuleNotFoundError

def plot_1d_spectrum(
        results1d: Defocus1DResults
):
    """Plot 1D power spectrum analysis results.
    
    Parameters
    ----------
    results1d : Defocus1DResults
        Results from 1D defocus estimation containing frequencies, power spectrum,
        background model, and CTF fitting results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: 1D Power Spectrum with background
    ax1 = axes[0, 0]
    freqs = results1d.frequencies_1d.detach().cpu().numpy()
    power_spec = results1d.powerspectrum_1d.detach().cpu().numpy()
    
    ax1.semilogy(freqs, power_spec, 'b-', alpha=0.7, label='Power Spectrum')
    
    # Add background model if available
    if results1d.background_model is not None and results1d.low_frequency_fit is not None:
        # Get fitting range
        fit_mask = (freqs >= results1d.low_frequency_fit) & (freqs <= results1d.high_frequency_fit)
        fit_freqs = freqs[fit_mask]
        
        # Evaluate background model
        x = torch.linspace(0, 1, steps=len(fit_freqs))
        background = torch.exp(results1d.background_model(x).squeeze()).detach().cpu().numpy()
        
        ax1.semilogy(fit_freqs, background, 'r--', alpha=0.8, label='Background Model')
        
        # Mark fitting range
        ax1.axvline(results1d.low_frequency_fit, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(results1d.high_frequency_fit, color='gray', linestyle=':', alpha=0.5)
        ax1.fill_betweenx(ax1.get_ylim(), results1d.low_frequency_fit, results1d.high_frequency_fit, 
                         alpha=0.1, color='gray', label='Fit Range')
    
    ax1.set_xlabel('Spatial Frequency (1/Å)')
    ax1.set_ylabel('Power')
    ax1.set_title('1D Power Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cross-correlation vs defocus
    if results1d.test_defoci is not None and results1d.cross_correlations is not None:
        ax2 = axes[0, 1]
        defoci = results1d.test_defoci.detach().cpu().numpy()
        correlations = results1d.cross_correlations.detach().cpu().numpy()
        
        ax2.plot(defoci, correlations, 'g-', linewidth=2)
        
        # Mark best defocus
        best_idx = correlations.argmax()
        best_defocus = defoci[best_idx]
        ax2.axvline(best_defocus, color='red', linestyle='--', 
                   label=f'Best Defocus: {best_defocus:.3f} μm')
        ax2.scatter(best_defocus, correlations[best_idx], color='red', s=100, zorder=5)
        
        ax2.set_xlabel('Defocus (μm)')
        ax2.set_ylabel('Cross-correlation')
        ax2.set_title('CTF Fitting Cross-correlation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Background-subtracted power spectrum in fit range
    if results1d.background_model is not None and results1d.low_frequency_fit is not None:
        ax3 = axes[1, 0]
        
        fit_mask = (freqs >= results1d.low_frequency_fit) & (freqs <= results1d.high_frequency_fit)
        fit_freqs = freqs[fit_mask]
        fit_power = power_spec[fit_mask]
        
        # Subtract background
        x = torch.linspace(0, 1, steps=len(fit_freqs))
        background = torch.exp(results1d.background_model(x).squeeze()).detach().cpu().numpy()
        corrected_power = fit_power - background
        
        # Normalize to 0-1 range
        corrected_power_min = corrected_power.min()
        corrected_power_max = corrected_power.max()
        corrected_power_normalized = (corrected_power - corrected_power_min) / (corrected_power_max - corrected_power_min)
        
        ax3.plot(fit_freqs, corrected_power_normalized, 'purple', linewidth=2, label='Background-subtracted')

        # Plot simulated CTF^2
        if results1d.ctf_model is not None:
            simulated_ctf2 = torch.sin(calculate_total_phase_shift(
                defocus_um=results1d.ctf_model.defocus_um,
                fftfreq_grid_angstrom_squared=fit_freqs**2,
                voltage_kv=results1d.ctf_model.voltage_kev,
                spherical_aberration_mm=results1d.ctf_model.spherical_aberration_mm,
                amplitude_contrast_fraction=results1d.ctf_model.amplitude_contrast_fraction,
                phase_shift_degrees=results1d.ctf_model.phase_shift_degrees
            )) ** 2

            ax3.plot(fit_freqs, simulated_ctf2, 'orange', linestyle='--', 
                     label='Simulated CTF^2')
        ax3.set_xlabel('Spatial Frequency (1/Å)')
        ax3.set_ylabel('Corrected Power')
        ax3.set_title('Background-subtracted Power Spectrum')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: CTF parameters summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Display CTF parameters as text
    ctf_info = [
        f"Defocus: {results1d.ctf_model.defocus_um:.3f} μm",
        f"Voltage: {results1d.ctf_model.voltage_kev:.1f} keV",
        f"Cs: {results1d.ctf_model.spherical_aberration_mm:.2f} mm",
        f"Amplitude contrast: {results1d.ctf_model.amplitude_contrast_fraction:.3f}",
        f"Phase shift: {results1d.ctf_model.phase_shift_degrees:.1f}°"
    ]
    
    if results1d.low_frequency_fit is not None:
        ctf_info.extend([
            "",
            f"Fit range: {results1d.low_frequency_fit:.3f} - {results1d.high_frequency_fit:.3f} 1/Å"
        ])
    
    ax4.text(0.1, 0.9, "CTF Parameters:", fontsize=14, fontweight='bold', 
             transform=ax4.transAxes, verticalalignment='top')
    
    for i, info in enumerate(ctf_info):
        ax4.text(0.1, 0.8 - i*0.08, info, fontsize=12, 
                transform=ax4.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()