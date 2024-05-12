import os
import cv2
import numpy as np
from glob import glob
from matplotlib import pyplot as plt

def compute_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

def apply_gradient_weight(image, gradient):
    normalized_gradient = cv2.normalize(gradient, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    adjusted_image = cv2.multiply(image.astype(np.float32), normalized_gradient)
    return adjusted_image

def compute_dft_and_spectrums_channel(channel):
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    phase_spectrum = np.angle(dft_shift[:, :, 0] + 1j * dft_shift[:, :, 1])
    return magnitude_spectrum, phase_spectrum

def inverse_dft_channel(magnitude_spectrum, phase_spectrum):
    magnitude = np.exp(magnitude_spectrum / 20)
    dft_complex = magnitude * np.exp(1j * phase_spectrum)
    dft_merged = cv2.merge([dft_complex.real.astype(np.float32), dft_complex.imag.astype(np.float32)])
    dft_reconstructed = np.fft.ifftshift(dft_merged)
    channel_reconstructed = cv2.idft(dft_reconstructed)
    channel_reconstructed = cv2.magnitude(channel_reconstructed[:, :, 0], channel_reconstructed[:, :, 1])
    return channel_reconstructed

def process_image(image_path):
    image = cv2.imread(image_path)
    reconstructed_image = np.zeros(image.shape, np.float32)
    for i in range(3):
        channel = image[:, :, i]
        gradient = compute_gradient(channel)
        weighted_channel = apply_gradient_weight(channel, gradient)
        magnitude_spectrum, phase_spectrum = compute_dft_and_spectrums_channel(weighted_channel)
        channel_reconstructed = inverse_dft_channel(magnitude_spectrum, phase_spectrum)
        channel_reconstructed_normalized = cv2.normalize(channel_reconstructed, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        reconstructed_image[:, :, i] = np.clip(channel_reconstructed_normalized, 0, 255).astype(np.uint8)
    return reconstructed_image

def save_reconstructed_image(image, new_path):
    cv2.imwrite(new_path, image)

def process_and_save_images(src_dir, dest_dir):
    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".png"):
                file_path = os.path.join(subdir, file)
                new_subdir = subdir.replace(src_dir, dest_dir)
                if not os.path.exists(new_subdir):
                    os.makedirs(new_subdir, exist_ok=True)
                new_file_path = os.path.join(new_subdir, file)
                image = process_image(file_path)
                save_reconstructed_image(image, new_file_path)


process_and_save_images('/opt/data/private/wangjuntong/code/FreqNet-DeepfakeDetection/dataset/GANGen-Detection', '/opt/data/private/wangjuntong/code/FreqNet-DeepfakeDetection/dataset/GANGen-Detection2')

# Note: The actual execution of this script should be performed in an appropriate environment where all dependencies are met, and file paths are accessible.
