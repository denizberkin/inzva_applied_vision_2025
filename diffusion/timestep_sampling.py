import numpy as np
import matplotlib.pyplot as plt

def logit_normal_density(t, m, s):
    t = np.clip(t, 1e-6, 1-1e-6)
    logit_t = np.log(t / (1 - t))
    density = (1 / (s * np.sqrt(2 * np.pi))) * (1 / (t * (1 - t))) * np.exp(-((logit_t - m)**2) / (2 * s**2))
    
    return density


def mode_density(t, s):
    t = np.clip(t, 1e-6, 1-1e-6)
    
    if s == 0:  # Uniform case
        return 1.0
    
    # approximate derivative
    def f_mode(u, s):
        return 1 - u - s * (np.cos(np.pi/2 * u)**2 - 1 + u)
    
    def inverse_f_mode(t, s):
        left, right = 0, 1
        for _ in range(20):  # 20 iterations should be enough for visualization purposes
            mid = (left + right) / 2
            if f_mode(mid, s) < t:
                left = mid
            else:
                right = mid
        return mid
    
    h = 1e-4
    u = inverse_f_mode(t, s)
    u_plus_h = inverse_f_mode(t + h, s)
    
    diff = (u_plus_h - u)
    if abs(diff) < 1e-10:  # If difference is too small
        derivative = 1.0  # Default value
    else:
        derivative = abs(1 / (diff / h))
    
    return derivative


def cosmap_density(t):
    return 2 / (np.pi - 2*np.pi*t + 2*np.pi*t**2)

def visualize_timestep_sampling(output_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    t = np.linspace(0.01, 0.99, 1000)
    params = [
        {"m": 0.0, "s": 1.0, "label": "LogitNormal(0.0, 1.0)", "color": "blue"},
        {"m": 0.5, "s": 0.6, "label": "LogitNormal(0.5, 0.6)", "color": "green"},
        {"m": 1.0, "s": 0.6, "label": "LogitNormal(1.0, 0.6)", "color": "red"},
        {"m": -0.5, "s": 1.0, "label": "LogitNormal(-0.5, 1.0)", "color": "purple"}
    ]
    
    uniform = np.ones_like(t)
    ax1.plot(t, uniform, 'k--', label="Uniform (baseline)", alpha=0.7)
    
    # plot logit-normal distributions
    for p in params:
        density = logit_normal_density(t, p["m"], p["s"])
        ax1.plot(t, density, label=p["label"], color=p["color"])
    
    mode_params = [
        {"s": -0.5, "label": "Mode(s=-0.5)", "color": "blue"},
        {"s": 0.0, "label": "Mode(s=0.0) (Uniform)", "color": "black"},
        {"s": 0.8, "label": "Mode(s=0.8)", "color": "green"},
        {"s": 1.25, "label": "Mode(s=1.25)", "color": "red"}
    ]
    
    for p in mode_params:
        density = [mode_density(tt, p["s"]) for tt in t]
        ax2.plot(t, density, label=p["label"], color=p["color"])
    
    # plot cosine mapping distribution
    density = cosmap_density(t)
    ax2.plot(t, density, 'c-', label="CosMap", linewidth=2)
    
    ax1.set_title("Logit-Normal Timestep Sampling Distributions")
    ax2.set_title("Mode Sampling & CosMap Distributions")
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Density π(t)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def visualize_timestep_weighting(output_path=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t = np.linspace(0.01, 0.99, 1000)
    # Calculate the weighting factor t/(1-t) * π(t) for different distributions
    
    # uniform weight, SNR
    uniform_weight = t / (1 - t)
    ax.plot(t, uniform_weight, 'k--', label="Uniform (original RF)", alpha=0.7)
    
    # Logit-normal weighting
    ln_params = [
        {"m": 0.0, "s": 1.0, "label": "LogitNormal(0.0, 1.0)", "color": "blue"},
        {"m": 0.5, "s": 0.6, "label": "LogitNormal(0.5, 0.6)", "color": "green"}
    ]
    
    for p in ln_params:
        ln_density = logit_normal_density(t, p["m"], p["s"])
        ln_weight = (t / (1 - t)) * ln_density
        # normalize, easier to visualize
        ln_weight = ln_weight / np.max(ln_weight) * np.max(uniform_weight)
        ax.plot(t, ln_weight, label=p["label"], color=p["color"])
    
    mode_params = [
        {"s": 0.8, "label": "Mode(s=0.8)", "color": "green"},
        {"s": 1.25, "label": "Mode(s=1.25)", "color": "red"}
    ]
    
    for p in mode_params:
        mode_dens = [mode_density(tt, p["s"]) for tt in t]
        mode_weight = (t / (1 - t)) * np.array(mode_dens)
        # normalize, easier to visualize
        mode_weight = mode_weight / np.max(mode_weight) * np.max(uniform_weight)
        ax.plot(t, mode_weight, label=p["label"], color=p["color"])
    
    cosmap_dens = cosmap_density(t)
    cosmap_weight = (t / (1 - t)) * cosmap_dens
    # normalize, easier to visualize
    cosmap_weight = cosmap_weight / np.max(cosmap_weight) * np.max(uniform_weight)
    ax.plot(t, cosmap_weight, 'c-', label="CosMap", linewidth=2)
    
    ax.set_title("Loss Weighting Factors for Different Timestep Distributions")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Weight Factor w(t) = t/(1-t) * π(t)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    visualize_timestep_sampling(output_path="outputs/timestep_sampling.png")
    visualize_timestep_weighting(output_path="outputs/timestep_weighting.png")