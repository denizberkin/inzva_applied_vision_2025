import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def visualize_snr_sampling():
    """
    Visualize the concept of Signal-to-Noise Ratio (SNR) sampling in diffusion models
    and how it's improved in rectified flow models.
    """
    # Create figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Time steps for visualization
    t = np.linspace(0.001, 0.999, 1000)
    
    # 1. Plot signal and noise levels for different diffusion methods
    ax = axs[0]
    
    # Standard diffusion (variance preserving) - simplified
    beta_start, beta_end = 0.0001, 0.02
    beta_t = beta_start + t * (beta_end - beta_start)
    alpha_t = 1 - beta_t
    alpha_bar_t = alpha_t  # Simplified
    
    # Signal and noise levels
    signal_diffusion = np.sqrt(alpha_bar_t)
    noise_diffusion = np.sqrt(1 - alpha_bar_t)
    
    # Rectified flow
    signal_rf = 1 - t
    noise_rf = t
    
    # Plot signal levels
    ax.plot(t, signal_diffusion, 'r-', label='Signal (Diffusion)')
    ax.plot(t, signal_rf, 'b-', label='Signal (Rectified Flow)')
    
    # Plot noise levels
    ax.plot(t, noise_diffusion, 'r--', label='Noise (Diffusion)')
    ax.plot(t, noise_rf, 'b--', label='Noise (Rectified Flow)')
    
    # Add titles and labels
    ax.set_title("Signal vs. Noise Levels")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True)
    
    # 2. Plot SNR (Signal-to-Noise Ratio)
    ax = axs[1]
    
    # Calculate SNR values
    snr_diffusion = (signal_diffusion / noise_diffusion)**2
    snr_rf = (signal_rf / noise_rf)**2
    
    # Plot in log scale
    ax.semilogy(t, snr_diffusion, 'r-', label='SNR (Diffusion)')
    ax.semilogy(t, snr_rf, 'b-', label='SNR (Rectified Flow)')
    
    # Add titles and labels
    ax.set_title("Signal-to-Noise Ratio (SNR)")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("SNR (log scale)")
    ax.legend()
    ax.grid(True)
    
    # 3. Plot sampling distributions optimized for SNR
    ax = axs[2]
    
    # Various sampling strategies from the paper
    
    # Calculate log-SNR values
    log_snr_diffusion = np.log(snr_diffusion)
    log_snr_rf = np.log(snr_rf)
    
    # LogitNormal(0.0, 1.0) - the best performer from the paper
    def logit_normal_density(t, m, s):
        t = np.clip(t, 1e-6, 1-1e-6)
        logit_t = np.log(t / (1 - t))
        return (1 / (s * np.sqrt(2 * np.pi))) * (1 / (t * (1 - t))) * np.exp(-((logit_t - m)**2) / (2 * s**2))
    
    ln_density = logit_normal_density(t, 0.0, 1.0)
    
    # EDM-style sampling (based on the paper's description)
    # EDM samples from a Gaussian in log-SNR space
    Pm, Ps = -1.2, 1.2  # Parameters from the paper
    log_snr_values = np.linspace(-10, 10, 1000)
    edm_density_log_snr = norm.pdf(log_snr_values, Pm, Ps)
    
    # Map densities to t-space
    # This is a simplification for visualization
    edm_density = np.interp(log_snr_diffusion, log_snr_values, edm_density_log_snr)
    
    # Normalize for better visualization
    ln_density = ln_density / np.max(ln_density)
    edm_density = edm_density / np.max(edm_density)
    
    # Uniform baseline (original RF)
    uniform = np.ones_like(t)
    uniform = uniform / np.max(uniform)
    
    # Plot the densities
    ax.plot(t, uniform, 'k--', label="Uniform (Original RF)", alpha=0.7)
    ax.plot(t, ln_density, 'b-', label="LogitNormal(0.0, 1.0)")
    ax.plot(t, edm_density, 'g-', label="EDM-style")
    
    # Add reference lines for log-SNR values
    ax2 = ax.twinx()
    ax2.plot(t, log_snr_diffusion, 'r-.', label="log-SNR (Diffusion)", alpha=0.4)
    ax2.plot(t, log_snr_rf, 'b-.', label="log-SNR (RF)", alpha=0.4)
    ax2.set_ylabel("log-SNR")
    ax2.legend(loc='upper right')
    
    # Add titles and labels
    ax.set_title("Optimized Sampling Distributions")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Sampling Density π(t)")
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Add a global title
    fig.suptitle("SNR-Based Samplers for Rectified Flow Models", fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    plt.show()

if __name__ == "__main__":
    visualize_snr_sampling()