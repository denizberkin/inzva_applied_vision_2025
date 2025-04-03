import numpy as np
import matplotlib.pyplot as plt


def visualize_trajectory_comparison(output_path=None):
    """
    Visualize the differences in trajectory between diffusion and rectified flow processes.
    
    Args:
        output_path: Optional path to save the visualization
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Time steps
    t = np.linspace(0, 1, 100)
    
    # Setup for standard diffusion trajectory
    beta_start, beta_end = 0.0001, 0.02
    beta_t = beta_start + t * (beta_end - beta_start)
    alpha_t = 1 - beta_t
    alpha_bar_t = alpha_t  # Simplified
    
    # In 2D space, show how a point moves from data to noise
    # Starting point (data) and ending point (noise)
    data_point = np.array([0.0, 0.0])
    noise_point = np.array([1.0, 1.0])
    
    # Calculate trajectories
    rf_trajectory = np.array([(1 - tt) * data_point + tt * noise_point for tt in t])
    
    # For diffusion, the trajectory curves based on the variance schedule
    diff_trajectory = np.array([np.sqrt(alpha_bar_tt) * data_point + 
                               np.sqrt(1 - alpha_bar_tt) * noise_point 
                               for alpha_bar_tt in alpha_bar_t])
    
    # Plot trajectories
    ax.plot(rf_trajectory[:, 0], rf_trajectory[:, 1], 'b-', linewidth=2, label='Rectified Flow (Straight)')
    ax.plot(diff_trajectory[:, 0], diff_trajectory[:, 1], 'r-', linewidth=2, label='Standard Diffusion (Curved)')
    
    # Mark data and noise points
    ax.scatter([data_point[0]], [data_point[1]], c='green', s=100, label='Data')
    ax.scatter([noise_point[0]], [noise_point[1]], c='orange', s=100, label='Noise')
    
    # Plot time markers
    for tt in [0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(tt * (len(t) - 1))
        ax.scatter([rf_trajectory[idx, 0]], [rf_trajectory[idx, 1]], c='blue', s=50)
        ax.scatter([diff_trajectory[idx, 0]], [diff_trajectory[idx, 1]], c='red', s=50)
        ax.text(rf_trajectory[idx, 0] + 0.02, rf_trajectory[idx, 1] + 0.02, f't={tt:.2f}', fontsize=9, color='blue')
        ax.text(diff_trajectory[idx, 0] - 0.1, diff_trajectory[idx, 1] - 0.02, f't={tt:.2f}', fontsize=9, color='red')
    
    # Label and grid
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Comparison of Trajectories: Rectified Flow vs Standard Diffusion')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    output_path = "outputs/trajectory_comparison.png"  # Optional output path
    visualize_trajectory_comparison(output_path)