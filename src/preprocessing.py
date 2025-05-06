import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_channels(img_partial_path, input_dir="../Dataset", output_dir="../Dataset", 
                  img_size=(512, 512), num_channels=8, save_numpy=True):
    """
    Processes a multi-channel TIFF image:
    - Loads each channel
    - Clips pixel values between 5th and 95th percentiles
    - Resizes to `img_size`
    - Normalizes to [0, 255]
    - Saves the stacked image as a NumPy array

    Parameters:
    - img_partial_path: str, base name of the image
    - input_dir: str, directory containing the original TIFF images
    - output_dir: str, directory to save the .npy output
    - img_size: tuple(int, int), target size (width, height)
    - num_channels: int, number of channels
    - save_numpy: bool, whether to save the result as a .npy file

    Returns:
    - img: np.ndarray, stacked image of shape (H, W, C)
    """
    image_channels = []

    for i in range(1, num_channels + 1):
        filename = f"{img_partial_path}-ch{i}sk1fk1fl1.tiff"
        filepath = os.path.join(input_dir, filename)
        img_resized = process_and_resize_image(img_size, filepath)
        image_channels.append(img_resized)

    img_stacked = np.stack(image_channels, axis=-1)

    if save_numpy:
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{img_partial_path}.npy"), img_stacked)

    return img_stacked

def process_and_resize_image(img_size, filepath):
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Image not found: {filepath}")

        # Contrast enhancement
    p5, p95 = np.percentile(img, [5, 95])
    img_clipped = np.clip(img, p5, p95)
    img_normalized = ((img_clipped - p5) / (p95 - p5) * 255).astype(np.uint8)

        # Resize
    img_resized = cv2.resize(img_normalized, img_size, interpolation=cv2.INTER_AREA)
    return img_resized

def load_channels(img_partial_path, channels = [1, 2, 5, 6, 7, 8]):
    """
    Loads each channel of the image as a separate NumPy array.

    Parameters:
    - img_partial_path: str, base name of the image
    - input_dir: str, directory containing the original TIFF images

    Returns:
    - img: np.ndarray, stacked image of shape (H, W, C)
    """

    images = []
    for i in channels:
        filename = f"{img_partial_path}-ch{i}sk1fk1fl1.tiff"
        img_resized = process_and_resize_image((512, 512), filename)
        images.append(img_resized)

    img_stacked = np.stack(images, axis=-1)

    return img_stacked


def visualize_image(img, cmap="gray"):
    """Visualizes all 8 channels in a horizontal layout."""
    fig, ax = plt.subplots(1, img.shape[-1], figsize=(20, 5))
    for i in range(img.shape[-1]):
        ax[i].imshow(img[:, :, i], cmap=cmap)
        ax[i].axis("off")
        ax[i].set_title(f"Channel {i+1}")
    plt.tight_layout()
    plt.show()