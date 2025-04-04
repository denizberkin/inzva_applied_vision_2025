import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import mark_boundaries, slic
from pathlib import Path
from fast_slic.avx2 import SlicAvx2


# https://pyradiomics.readthedocs.io/en/latest/features.html
# try to extract features from pseudo-generated mask and original image
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
        raise ValueError(f"Could not read image: {image_path}")
    
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
    
    # Apply SLIC, either using fast_slic or skimage
    # slic_segments = slic(filtered_image, n_segments=n_segments, compactness=compactness, 
    #                      mask=full_mask, max_num_iter=10)
    
    min_size_factor=0.25
    convert_to_lab=True 
    slic = SlicAvx2(
        num_components=n_segments,
        compactness=compactness,
        min_size_factor=min_size_factor,
        convert_to_lab=convert_to_lab
    )
    
    slic_segments = slic.iterate(filtered_image)
    
    # Process with the requested functions
    mean_intensity_image = slicMeanIntensityImage(filtered_image, slic_segments)
    labels = getSlicTrainLabels(mean_intensity_image, slic_segments, quantileDist=quantile_dist)
    marked_image = markSegments(slic_segments, labels)
    
    return original_image, resized_image, slic_segments, mean_intensity_image, marked_image

def visualize_and_save_results(results, output_path):
    original_image, resized_image, slic_segments, mean_intensity_image, marked_image = results
    
    plt.figure(figsize=(16, 12))
    
    # Original image
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
    
    # Mean intensity image
    plt.subplot(2, 2, 3)
    plt.imshow(mean_intensity_image, cmap='gray')
    plt.title('Mean Intensity Image')
    plt.axis('off')
    
    # Marked mask
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
                apply_filter=apply_filter
            )
            
            # Create the output file path
            output_file = output_path / f"{filename}_slic_results.png"
            
            # Visualize and save results
            visualize_and_save_results(results, output_file)
            
            print(f"Processed: {png_file} -> {output_file}")
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {png_file}: {e}")
    
    return processed_count


if __name__ == "__main__":
    input_folder = "images"
    output_folder = "outputs"
    
    num_processed = process_image_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        n_segments=500,
        compactness=1,
        sigma=2.0,
        resize_factor=1.0,
        quantile_dist=0.5,
        apply_filter=True
    )
    
    print(f"Processing complete. {num_processed} images processed successfully.")