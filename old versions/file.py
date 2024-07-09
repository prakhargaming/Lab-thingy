import numpy as np
import cv2
from scipy import signal
from scipy import fftpack
from matplotlib import pyplot as plt

def wiener_deconvolution(input_image, target_image, noise_level=0.1):
    # Compute FFT of input and target images
    input_fft = fftpack.fft2(input_image)
    target_fft = fftpack.fft2(target_image)
    
    # Estimate the filter in frequency domain
    filter_fft = np.conj(input_fft) * target_fft / (np.abs(input_fft)**2 + noise_level**2)
    
    # Convert filter back to spatial domain
    kernel = np.real(fftpack.ifft2(filter_fft))
    
    # Normalize kernel
    kernel = kernel / np.sum(kernel)
    
    return kernel

def apply_kernel(image, kernel):
    return signal.convolve2d(image, kernel, mode='same', boundary='wrap')

# Load images
input_image = cv2.imread('Prostate.jpg', cv2.IMREAD_GRAYSCALE)
target_image = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure images are the same size
min_shape = np.minimum(input_image.shape, target_image.shape)
input_image = input_image[:min_shape[0], :min_shape[1]]
target_image = target_image[:min_shape[0], :min_shape[1]]

# Compute the kernel using Wiener deconvolution
kernel = wiener_deconvolution(input_image, target_image)

# Apply the kernel to the input image
result_image = apply_kernel(input_image, kernel)

# Display results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(input_image, cmap='gray'), plt.title('Input Image')
plt.subplot(132), plt.imshow(result_image, cmap='gray'), plt.title('Result Image')
plt.subplot(133), plt.imshow(target_image, cmap='gray'), plt.title('Target Image')
plt.tight_layout()
plt.show()

# Display the kernel
plt.figure(figsize=(5, 5))
plt.imshow(kernel, cmap='gray')
plt.title('Computed Kernel')
plt.colorbar()
plt.show()