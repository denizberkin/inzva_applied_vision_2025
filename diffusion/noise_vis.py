import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image


def add_noise(x, t_step, max_steps=1000):
    """
    Add noise to the input tensor according to the diffusion process.
    
    Args:
        x: Input tensor [B, C, H, W]
        t_step: Current timestep (0 to max_steps)
        max_steps: Maximum number of diffusion steps
        
    Returns:
        Noised tensor at timestep t
    """
    # Generate noise
    noise = torch.randn_like(x)
    
    # Linear beta schedule
    beta_start = 0.00085
    beta_end = 0.0120
    
    # Convert timestep to relative position in schedule
    t_scaled = t_step / max_steps
    
    # Calculate beta_t for the current timestep
    beta_t = beta_start + t_scaled * (beta_end - beta_start)
    
    # Convert to alpha_t and cumulative product
    alphas = 1.0 - beta_t
    alphas_cumprod = alphas
    for i in range(1, int(t_step)):
        beta_i = beta_start + (i / max_steps) * (beta_end - beta_start)
        alpha_i = 1.0 - beta_i
        alphas_cumprod *= alpha_i
    
    # Convert to torch tensors for the operations
    sqrt_alphas_cumprod = torch.tensor(np.sqrt(alphas_cumprod), dtype=x.dtype, device=x.device)
    sqrt_one_minus_alphas_cumprod = torch.tensor(np.sqrt(1.0 - alphas_cumprod), dtype=x.dtype, device=x.device)
    
    # Calculate noisy sample
    noised_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
    
    return noised_x, noise

def visualize_diffusion_process(image_path, output_path=None):
    """
    visualize diffusion process alike DDPM.
    """
    img = Image.open(image_path).convert('RGB')
    
    if max(img.size) > 512:
        transform = T.Resize(512, antialias=True)
        img = transform(img)
    
    transform = T.Compose([T.ToTensor(),])
    
    # add batch
    x = transform(img).unsqueeze(0)
    
    max_steps = 1000
    
    x_mid, _ = add_noise(x, max_steps // 8, max_steps)  # t=250 (quarter of diffusion)
    x_full, _ = add_noise(x, max_steps - 1, max_steps)  # t=999 (nearly pure noise)
    
    # Convert back to PIL images for display
    original_img = to_pil_image(x.squeeze(0).clip(0, 1))
    halfway_img = to_pil_image(x_mid.squeeze(0).clip(0, 1))
    fully_noised_img = to_pil_image(x_full.squeeze(0).clip(0, 1))
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title(f'Original Image (t=0)')
    axes[0].axis('off')
    
    axes[1].imshow(halfway_img)
    axes[1].set_title(f'Halfway Noised (t={max_steps // 8})')
    axes[1].axis('off')
    
    axes[2].imshow(fully_noised_img)
    axes[2].set_title(f'Nearly Pure Noise (t={max_steps - 1})')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
    
    plt.show()

if __name__ == "__main__":
    im_name = "bmo_space"
    image_path = f"inputs/{im_name}.png"
    output_path = f"outputs/{im_name}_noised.png"  # Optional, set to None to not save
    
    visualize_diffusion_process(image_path, output_path)