import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.gridspec as gridspec

np.random.seed(42)


def add_noise_diffusion(x, t, beta_start=0.0001, beta_end=0.02, noise=None):
    if noise is None:
        noise = np.random.randn(*x.shape)
    
    beta_t = beta_start + t * (beta_end - beta_start)
    
    alpha_t = 1 - beta_t
    alpha_bar_t = alpha_t
    
    # add noise
    sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
    
    noised_x = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
    
    return noised_x


def add_noise_rectified_flow(x, t, noise=None):
    if noise is None:
        noise = np.random.randn(*x.shape)
    
    # rectified flow
    noised_x = (1 - t) * x + t * noise
    
    return noised_x

def visualize_noise_processes(image_path, output_path=None):
    img = Image.open(image_path).convert('RGB')
    if max(img.size) > 512:
        img = img.resize((512, 512))
    
    # norm img and generate noise
    x = np.array(img).astype(np.float32) / 255.0
    noise = np.random.randn(*x.shape)
    
    # visualize at given time steps
    time_steps = [0, 0.25, 0.5, 0.75, 1.0]
    n_steps = len(time_steps)
    
    fig = plt.figure(figsize=(5 * n_steps, 10))
    gs = gridspec.GridSpec(2, n_steps)
    
    # vis standard diffusion
    for i, t in enumerate(time_steps):
        noised_x_diffusion = add_noise_diffusion(x, t, noise=noise)
        noised_x_diffusion = np.clip(noised_x_diffusion, 0, 1)
        
        ax = plt.subplot(gs[0, i])
        ax.imshow(noised_x_diffusion)
        ax.set_title(f'Standard Diffusion (t={t:.2f})')
        ax.axis('off')
    
    # vis rectified flow
    for i, t in enumerate(time_steps):
        noised_x_rf = add_noise_rectified_flow(x, t, noise=noise)
        noised_x_rf = np.clip(noised_x_rf, 0, 1)
        
        ax = plt.subplot(gs[1, i])
        ax.imshow(noised_x_rf)
        ax.set_title(f'Rectified Flow (t={t:.2f})')
        ax.axis('off')
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # sorry for a bit slow implementation, numpy and torch versions got a bit mixed up
    image_path = "inputs/bmo_space.png"  # Replace with your image path
    output_path = "outputs/noise_comparison.png"  # Optional output path
    visualize_noise_processes(image_path, output_path)