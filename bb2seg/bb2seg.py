""" 
If you have problems installing fast_slic [there are some issues with dependencies on windows-based systems],
you should comment & un-comment the lines in the code below.


# https://pyradiomics.readthedocs.io/en/latest/features.html
# try to extract features from pseudo-generated mask and original image


# slic algorithm short-papers & fast_slic implementation:
fast_slic repository: https://github.com/Algy/fast-slic: reasonable explanation of the algorithm, easy to follow.
original research [locked behind institution access (can check from #achante2012.pdf)]: https://ieeexplore.ieee.org/document/6205760
maskSLIC Local Pathology Characterisation in Medical Images: https://arxiv.org/pdf/1606.09518
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries, slic
from pathlib import Path

try:
    from fast_slic.avx2 import SlicAvx2
    FAST_SLIC_AVAILABLE = True
except ImportError:
    FAST_SLIC_AVAILABLE = False


## TESTING
def find_optimal_quantile(
    image: np.ndarray, 
    slicMask: np.ndarray, 
    num_steps: int = 10) -> float:
    """
    Find the optimal quantile value for the image based on intensity distribution.
    Hope this works out for me at one point.
    """
    if len(image.shape) == 3:
        image = image[..., 0]
    
    # mean intensity image
    meanImage = slicMeanIntensityImage(image, slicMask)
    
    # init
    best_quantile = 0.5
    max_variance = 0
    
    # try quantiles
    for q in np.linspace(0.3, 0.7, num_steps):
        # Get superpixel labels based on this quantile
        labels = getSlicTrainLabels(meanImage, slicMask, quantileDist=q)
        
        binary_mask = np.zeros_like(slicMask, dtype=np.uint8)
        for sliclabel in np.unique(slicMask):
            if labels[sliclabel]:
                binary_mask[slicMask == sliclabel] = 1
        
        # variance between foreground and background
        if np.sum(binary_mask) > 0 and np.sum(binary_mask) < binary_mask.size:
            fg_mean = np.mean(image[binary_mask == 1])
            bg_mean = np.mean(image[binary_mask == 0])
            variance = (fg_mean - bg_mean) ** 2
            
            if variance > max_variance:
                max_variance = variance
                best_quantile = q
    
    return best_quantile


def slicMeanIntensityImage(image, slicMask):
    if len(image.shape) == 3:
        image = image[..., 0]
    meanImage = np.zeros_like(image)
    for slicLabel in np.unique(slicMask):
        meanImage[slicMask == slicLabel] = np.mean(image[slicMask == slicLabel])
    return meanImage


def getSlicTrainLabels(meanImage, slicMask, quantileDist=0.5):
    labels = {}
    
    q = np.quantile(meanImage, quantileDist)

    for slicLabel in np.unique(slicMask):
        labels[slicLabel] = (np.mean(meanImage[slicMask == slicLabel]) > q)

    return labels


def markSegments(slicMask, labels):
    marked_image = np.zeros_like(slicMask, dtype=np.uint8)

    for sliclabel in np.unique(slicMask):
        if labels[sliclabel]:
            marked_image[slicMask == sliclabel] = 255

    return marked_image


def process_single_image(image_path, n_segments=200, compactness=10, sigma=2.0, 
                         resize_factor=1.0, quantile_dist=0.5, apply_filter=True):
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}, might want to check path or opencv!")
    
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    if resize_factor != 1.0:
        resized_image = cv2.resize(original_image, None, fx=resize_factor, fy=resize_factor, 
                                  interpolation=cv2.INTER_CUBIC)
    else:
        resized_image = original_image.copy()
    
    if apply_filter:
        filtered_image = cv2.GaussianBlur(resized_image, (5, 5), sigmaX=sigma, sigmaY=sigma)
    else:
        filtered_image = resized_image.copy()
    
    full_mask = np.ones(filtered_image.shape[:2], dtype=np.uint8)
    
    # apply SLIC, either using fast_slic or skimage
    # check if fast_slic library is i
    if FAST_SLIC_AVAILABLE:
        min_size_factor=0.25
        convert_to_lab=True 
        slic = SlicAvx2(
            num_components=n_segments,
            compactness=compactness,
            min_size_factor=min_size_factor,
            convert_to_lab=convert_to_lab
        )
        slic_segments = slic.iterate(filtered_image)
    else:
        slic_segments = slic(filtered_image, n_segments=n_segments, compactness=compactness, 
                             mask=full_mask, max_num_iter=10)
    
    # main post-processing pipeline after superpixel calculation 
    mean_intensity_image = slicMeanIntensityImage(filtered_image, slic_segments)
    labels = getSlicTrainLabels(mean_intensity_image, slic_segments, quantileDist=quantile_dist)
    marked_image = markSegments(slic_segments, labels)
    
    return original_image, resized_image, slic_segments, mean_intensity_image, marked_image


def visualize_and_save_results(results, output_path):
    original_image, resized_image, slic_segments, mean_intensity_image, marked_image = results
    
    plt.figure(figsize=(16, 12))
    
    # original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # SLIC boundaries
    plt.subplot(2, 2, 2)
    boundaries = mark_boundaries(resized_image, slic_segments, color=(1, 0.5, 0), 
                                outline_color=None, mode='subpixel')
    plt.imshow(boundaries)
    plt.title('SLIC Boundaries')
    plt.axis('off')
    
    # mean intensity image
    plt.subplot(2, 2, 3)
    plt.imshow(mean_intensity_image, cmap='gray')
    plt.title('Mean Intensity Image')
    plt.axis('off')
    
    # marked mask
    plt.subplot(2, 2, 4)
    plt.imshow(marked_image, cmap='gray')
    plt.title('Marked Mask')
    plt.axis('off')
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def process_image_folder(input_folder, output_folder, n_segments=200, compactness=10, sigma=2.0, 
                         resize_factor=1.0, quantile_dist=0.5, apply_filter=True):
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    png_files = list(input_path.glob('*.png'))
    
    processed_count = 0
    for png_file in png_files:
        try:
            filename = png_file.stem
            
            results = process_single_image(
                image_path=png_file,
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
                resize_factor=resize_factor,
                quantile_dist=quantile_dist,
                apply_filter=apply_filter,
            )
            
            output_file = output_path / f"{filename}_slic_results.png"
            visualize_and_save_results(results, output_file)
            print(f"Processed: {png_file} -> {output_file}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    return processed_count


if __name__ == "__main__":
    # retina images operate on inverse of what mean_intensity is intended, 
    # so I'll need to parametrize and add a mechanism to calculate based on that as well
    input_folder = "images"
    output_folder = "outputs"
    
    num_processed = process_image_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        n_segments=500,
        compactness=1,
        sigma=2.0,
        resize_factor=1.0,
        apply_filter=True,
        quantile_dist=0.99,  # value to be tuned while creating absolute mask (bottom right in the plot)
        # higher quantile dist is used for brigher or intenser image modalities
    )
    
    print(f"Processing complete. {num_processed} images processed successfully.")