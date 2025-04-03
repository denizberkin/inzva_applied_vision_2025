import numpy as np
import matplotlib.pyplot as plt

def logit_normal_density(t, m, s):
    """
    Calculate the density of the logit-normal distribution at point t.
    
    Args:
        t: Point at which to evaluate the density (between 0 and 1)
        m: Location parameter
        s: Scale parameter
    
    Returns:
        Density value at t
    """
    # Avoid numerical issues at boundaries
    t = np.clip(t, 1e-6, 1-1e-6)
    
    # Logit function
    logit_t = np.log(t / (1 - t))
    
    # Density formula
    density = (1 / (s * np.sqrt(2 * np.pi))) * (1 / (t * (1 - t))) * np.exp(-((logit_t - m)**2) / (2 * s**2))
    
    return density

def mode_density(t, s):
    """
    Calculate the density of the mode-sampling distribution at point t.
    
    Args:
        t: Point at which to evaluate the density (between 0 and 1)
        s: Scale parameter controlling the shape (-1 to 1.75)
    
    Returns:
        Density value at t
    """
    # Use the derivative of the inverse mapping
    # This is a simplified implementation
    
    # Avoid numerical issues at boundaries
    t = np.clip(t, 1e-6, 1-1e-6)
    
    # The formula from the paper (equation 20)
    if s == 0:  # Uniform case
        return 1.0
    
    # Numerical approximation of the derivative
    def f_mode(u, s):
        return 1 - u - s * (np.cos(np.pi/2 * u)**2 - 1 + u)
    
    # Compute inverse using a simple binary search
    # (This is just for visualization, not efficient for actual sampling)
    def inverse_f_mode(t, s):
        left, right = 0, 1
        for _ in range(20):  # 20 iterations should be enough for visualization purposes
            mid = (left + right) / 2
            if f_mode(mid, s) < t:
                left = mid
            else:
                right = mid
        return mid
    
    # Approximate derivative using finite differences
    h = 1e-4
    u = inverse_f_mode(t, s)
    u_plus_h = inverse_f_mode(t + h, s)
    
    # Avoid division by zero
    diff = (u_plus_h - u)
    if abs(diff) < 1e-10:  # If difference is too small
        derivative = 1.0  # Default value
    else:
        derivative = abs(1 / (diff / h))
    
    return derivative

def cosmap_density(t):
    """
    Calculate the density of the cosine mapping distribution at point t.
    
    Args:
        t: Point at which to evaluate the density (between 0 and 1)
    
    Returns:
        Density value at t
    """
    # The formula from the paper (equation 22)
    return 2 / (np.pi - 2*np.pi*t + 2*np.pi*t**2)

def visualize_timestep_sampling():
    """
    Visualize different timestep sampling strategies for rectified flow
    and their impact on weighting training data.
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Time points for evaluation
    t = np.linspace(0.01, 0.99, 1000)
    
    # Plot logit-normal distributions with different parameters
    params = [
        {"m": 0.0, "s": 1.0, "label": "LogitNormal(0.0, 1.0)", "color": "blue"},
        {"m": 0.5, "s": 0.6, "label": "LogitNormal(0.5, 0.6)", "color": "green"},
        {"m": 1.0, "s": 0.6, "label": "LogitNormal(1.0, 0.6)", "color": "red"},
        {"m": -0.5, "s": 1.0, "label": "LogitNormal(-0.5, 1.0)", "color": "purple"}
    ]
    
    # Uniform distribution baseline
    uniform = np.ones_like(t)
    ax1.plot(t, uniform, 'k--', label="Uniform (baseline)", alpha=0.7)
    
    # Plot logit-normal distributions
    for p in params:
        density = logit_normal_density(t, p["m"], p["s"])
        ax1.plot(t, density, label=p["label"], color=p["color"])
    
    # Plot mode sampling distributions
    mode_params = [
        {"s": -0.5, "label": "Mode(s=-0.5)", "color": "blue"},
        {"s": 0.0, "label": "Mode(s=0.0) (Uniform)", "color": "black"},
        {"s": 0.8, "label": "Mode(s=0.8)", "color": "green"},
        {"s": 1.25, "label": "Mode(s=1.25)", "color": "red"}
    ]
    
    # Plot mode distributions
    for p in mode_params:
        density = [mode_density(tt, p["s"]) for tt in t]
        ax2.plot(t, density, label=p["label"], color=p["color"])
    
    # Plot cosine mapping distribution
    density = cosmap_density(t)
    ax2.plot(t, density, 'c-', label="CosMap", linewidth=2)
    
    # Add titles and labels
    ax1.set_title("Logit-Normal Timestep Sampling Distributions")
    ax2.set_title("Mode Sampling & CosMap Distributions")
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Timestep t")
        ax.set_ylabel("Density π(t)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

def visualize_timestep_weighting():
    """
    Visualize how different timestep sampling strategies affect 
    the weighting of training data in the loss function.
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Time points for evaluation
    t = np.linspace(0.01, 0.99, 1000)
    
    # Calculate the weighting factor t/(1-t) * π(t) for different distributions
    
    # Uniform weighting (baseline RF)
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
        # Normalize for better visualization
        ln_weight = ln_weight / np.max(ln_weight) * np.max(uniform_weight)
        ax.plot(t, ln_weight, label=p["label"], color=p["color"])
    
    # Mode sampling weighting
    mode_params = [
        {"s": 0.8, "label": "Mode(s=0.8)", "color": "green"},
        {"s": 1.25, "label": "Mode(s=1.25)", "color": "red"}
    ]
    
    for p in mode_params:
        mode_dens = [mode_density(tt, p["s"]) for tt in t]
        mode_weight = (t / (1 - t)) * np.array(mode_dens)
        # Normalize for better visualization
        mode_weight = mode_weight / np.max(mode_weight) * np.max(uniform_weight)
        ax.plot(t, mode_weight, label=p["label"], color=p["color"])
    
    # CosMap weighting
    cosmap_dens = cosmap_density(t)
    cosmap_weight = (t / (1 - t)) * cosmap_dens
    # Normalize for better visualization
    cosmap_weight = cosmap_weight / np.max(cosmap_weight) * np.max(uniform_weight)
    ax.plot(t, cosmap_weight, 'c-', label="CosMap", linewidth=2)
    
    # Add annotations
    ax.annotate('Higher weighting in intermediate steps', 
                xy=(0.5, 3), xytext=(0.6, 5),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    ax.annotate('Original RF weights\ntoo heavily on t≈1', 
                xy=(0.9, 8), xytext=(0.7, 10),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Add titles and labels
    ax.set_title("Loss Weighting Factors for Different Timestep Distributions")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Weight Factor w(t) = t/(1-t) * π(t)")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 12)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_timestep_sampling()
    visualize_timestep_weighting()