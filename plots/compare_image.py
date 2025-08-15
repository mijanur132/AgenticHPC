import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.util import img_as_float
from skimage.transform import resize

def load_and_preprocess(path):
    img = imread(path)
    if img.shape[-1] == 4:
        img = img[..., :3]  # Remove alpha channel
    img = img_as_float(img)
    return rgb2gray(img)

def compare_images(img_path_a, img_path_b):
    gray_a = load_and_preprocess(img_path_a)
    gray_b = load_and_preprocess(img_path_b)

    # Resize to match shapes
    target_shape = min(gray_a.shape, gray_b.shape)
    gray_a = resize(gray_a, target_shape, anti_aliasing=True)
    gray_b = resize(gray_b, target_shape, anti_aliasing=True)

    # Compute metrics
    ssim_val = ssim(gray_a, gray_b, data_range=1.0)
    mse_val = mse(gray_a, gray_b)
    psnr_val = psnr(gray_a, gray_b, data_range=1.0)

    # Compute and show visual diff
    diff_map = np.abs(gray_a - gray_b)
    plt.figure(figsize=(10, 4))
    plt.imshow(diff_map, cmap='hot')
    plt.colorbar(label='Absolute Difference')
    plt.title('Visual Difference Map (Image A - Image B)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Print metrics
    print(f"SSIM:  {ssim_val:.4f}")
    print(f"MSE:   {mse_val:.6f}")
    print(f"PSNR:  {psnr_val:.2f} dB")

if __name__ == "__main__":
    # Set your file paths here
    img_path_a = "a.png"
    img_path_b = "b.png"
    compare_images(img_path_a, img_path_b)
