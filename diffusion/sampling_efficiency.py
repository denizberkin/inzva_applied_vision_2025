import numpy as np
import matplotlib.pyplot as plt


def visualize_sampling_steps_comparison(output_path=None):
    # num steps
    steps = np.array([5, 10, 25, 50, 100])
    
    # illustrative FID scores
    diffusion_fid = np.array([90, 65, 38, 22, 19])
    rectified_flow_fid = np.array([45, 30, 22, 20, 19])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, diffusion_fid, 'ro-', linewidth=2, markersize=10, label='Standard Diffusion')
    ax.plot(steps, rectified_flow_fid, 'bo-', linewidth=2, markersize=10, label='Rectified Flow')
    
    ax.set_xlabel('Number of Sampling Steps')
    ax.set_ylabel('FID Score (Lower is Better)')
    ax.set_title('Sampling Efficiency: Rectified Flow vs Standard Diffusion')
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


def visualize_path_length_comparison(output_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    t = np.linspace(0, 1, 1000)
    beta_start, beta_end = 0.0001, 0.02
    beta_t = beta_start + t * (beta_end - beta_start)
    alpha_t = 1 - beta_t
    alpha_bar_t = alpha_t
    
    data_point = np.array([0.0, 0.0])
    noise_point = np.array([1.0, 1.0])
    
    rf_trajectory = np.array([(1 - tt) * data_point + tt * noise_point for tt in t])
    
    diff_trajectory = np.array([np.sqrt(alpha_bar_tt) * data_point + 
                               np.sqrt(1 - alpha_bar_tt) * noise_point 
                               for alpha_bar_tt in alpha_bar_t])
    
    def path_length(trajectory):
        """ sum segment lengths"""
        segments = np.diff(trajectory, axis=0)
        segment_lengths = np.sqrt(np.sum(segments**2, axis=1))
        return np.sum(segment_lengths)
    
    rf_length = path_length(rf_trajectory)
    diff_length = path_length(diff_trajectory)
    
    ax.plot(rf_trajectory[:, 0], rf_trajectory[:, 1], 'b-', linewidth=3, 
            label=f'Rectified Flow (Length: {rf_length:.3f})')
    ax.plot(diff_trajectory[:, 0], diff_trajectory[:, 1], 'r-', linewidth=3, 
            label=f'Standard Diffusion (Length: {diff_length:.3f})')
    
    # fewer points for rectified flow is enough
    rf_samples = 5
    diff_samples = 15
    
    rf_sample_indices = np.linspace(0, len(t)-1, rf_samples, dtype=int)
    diff_sample_indices = np.linspace(0, len(t)-1, diff_samples, dtype=int)
    
    # Plot sample points
    ax.scatter(rf_trajectory[rf_sample_indices, 0], rf_trajectory[rf_sample_indices, 1], 
               color='blue', s=100, zorder=10, label=f'RF Samples ({rf_samples} steps)')
    ax.scatter(diff_trajectory[diff_sample_indices, 0], diff_trajectory[diff_sample_indices, 1], 
               color='red', s=100, zorder=10, label=f'Diffusion Samples ({diff_samples} steps)')
    
    # Mark data and noise points
    ax.scatter([data_point[0]], [data_point[1]], c='green', s=200, zorder=11, label='Data')
    ax.scatter([noise_point[0]], [noise_point[1]], c='orange', s=200, zorder=11, label='Noise')
    
    # Label and grid
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title('Path Length Comparison: Rectified Flow vs Standard Diffusion')
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Add a ratio text
    ratio = diff_length / rf_length
    ax.text(0.5, -0.1, f"Path Length Ratio (Diffusion/RF): {ratio:.2f}x", 
            ha='center', fontsize=12, transform=ax.transAxes, 
            bbox=dict(facecolor='yellow', alpha=0.2))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    output_path1 = "outputs/sampling_efficiency.png"
    visualize_sampling_steps_comparison(output_path1)
    
    output_path2 = "outputs/path_length.png"
    visualize_path_length_comparison(output_path2)