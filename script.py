import numpy as np
import cv2
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt

def gaussian_filter(size, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g / g.sum()

def apply_kernel(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return signal.convolve2d(image, kernel, mode='same', boundary='wrap')


def image_to_frequency_domain(image):
    """
    Convert an image to its frequency domain representation.
    
    Parameters:
    image (numpy.ndarray): Input image as a 2D numpy array.
    
    Returns:
    tuple: (fft_shifted, magnitude_spectrum)
        - fft_shifted: The shifted Fourier Transform result
        - magnitude_spectrum: The magnitude spectrum of the frequency domain
    """
    # Apply 2D FFT
    fft_result = np.fft.fft2(image)
    
    # Shift the zero-frequency component to the center of the spectrum
    fft_shifted = np.fft.fftshift(fft_result)
    
    # Calculate the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))
    
    return fft_shifted, magnitude_spectrum

def plot_image_and_spectrum(image, magnitude_spectrum):
    """
    Plot the original image and its frequency domain representation.
    
    Parameters:
    image (numpy.ndarray): Original input image
    magnitude_spectrum (numpy.ndarray): Magnitude spectrum of the frequency domain
    """
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

kernal_size = 100

hi1 = gaussian_filter(kernal_size, 0.3)
hi2 = gaussian_filter(kernal_size, 1)
hi3 = gaussian_filter(kernal_size, 0.1)

hi = hi1 + hi2 + hi3

plt.imshow(hi, cmap="gray")
