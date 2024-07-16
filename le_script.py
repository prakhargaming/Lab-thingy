# %% [markdown]
# ## Imports

# %%
import cv2
import cupy as cp
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from cupyx.scipy.signal import fftconvolve

# %% [markdown]
# ## Helper Functions

# %%
def convolve_arrays(array1: np.ndarray, array2: np.ndarray, mode='same') -> cp.ndarray:
    cupy_array1 = cp.asarray(array1)
    cupy_array2 = cp.asarray(array2)
    return fftconvolve(cupy_array1, cupy_array2, mode=mode)

# %%
def gaussian_filter(size, sigma):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x*x + y*y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g / g.sum()

# %%
def MSE(arr1, arr2):
    mse = cp.mean((arr1 - arr2) ** 2)
    return mse

# %%
plt.imshow(gaussian_filter(21, 2), cmap="gray")

# %% [markdown]
# ## Algo

# %%
ssim_kernel_div = 0
ssim_sigma_div = 0
ssim_best_div = -float('inf')
ssim_image_div = 0

mse_kernel_div = 0
mse_sigma_div = 0
mse_best_div = float('inf')
mse_image_div = 0

ssim_kernel_sub = 0
ssim_sigma_sub = 0
ssim_best_sub = -float('inf')
ssim_image_sub = 0

mse_kernel_sub = 0
mse_sigma_sub = 0
mse_best_sub = float('inf')
mse_image_sub = 0

# %%
input_image = cv2.imread('Prostate.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
target_image = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255.0
input_image_cp = cp.array(input_image)
target_image_cp = cp.array(target_image)

print(ssim(input_image, target_image, data_range=1.0))

# %%
for kernel_size in range(11, 55, 2):
    for sigma in np.arange(0.1, 5, 0.1):
        gaussian = cp.asarray(gaussian_filter(kernel_size, sigma))
        image_le_temp = fftconvolve(input_image_cp, gaussian, mode='same')

        # Normalize subtraction result
        substract = cp.subtract(input_image_cp, image_le_temp)
        substract_min, substract_max = cp.min(substract), cp.max(substract)
        substract_normalized = 0.7 * (substract - substract_min) / (substract_max - substract_min)

        # Normalize division result
        with cp.errstate(divide='ignore', invalid='ignore'):
            division = cp.divide(input_image_cp, image_le_temp)
            division[image_le_temp == 0] = 0
        division_min, division_max = cp.min(division), cp.max(division)
        division_normalized = 0.7 * (division - division_min) / (division_max - division_min)

        ssim_value_div = ssim(cp.asnumpy(division_normalized), cp.asnumpy(target_image_cp), data_range=0.7)
        mse_value_div = MSE(division_normalized, target_image_cp)
        
        ssim_value_sub = ssim(cp.asnumpy(substract_normalized), cp.asnumpy(target_image_cp), data_range=0.7)
        mse_value_sub = MSE(substract_normalized, target_image_cp)

        if ssim_value_div > ssim_best_div:
            ssim_kernel_div, ssim_sigma_div, ssim_best_div = kernel_size, sigma, ssim_value_div
            print(f"New best SSIM (division): {ssim_value_div:.4f} (kernel: {kernel_size}, sigma: {sigma:.2f})")

        if mse_value_div < mse_best_div:
            mse_kernel_div, mse_sigma_div, mse_best_div = kernel_size, sigma, mse_value_div
            print(f"New best MSE (division): {mse_value_div:.4f} (kernel: {kernel_size}, sigma: {sigma:.2f})")

        if ssim_value_sub > ssim_best_sub:
            ssim_kernel_sub, ssim_sigma_sub, ssim_best_sub = kernel_size, sigma, ssim_value_sub
            print(f"New best SSIM (subtraction): {ssim_value_sub:.4f} (kernel: {kernel_size}, sigma: {sigma:.2f})")

        if mse_value_sub < mse_best_sub:
            mse_kernel_sub, mse_sigma_sub, mse_best_sub = kernel_size, sigma, mse_value_sub
            print(f"New best MSE (subtraction): {mse_value_sub:.4f} (kernel: {kernel_size}, sigma: {sigma:.2f})")

print(f"Final best SSIM (division): {ssim_best_div:.4f} (kernel: {ssim_kernel_div}, sigma: {ssim_sigma_div:.2f})")
print(f"Final best MSE (division): {mse_best_div:.4f} (kernel: {mse_kernel_div}, sigma: {mse_sigma_div:.2f})")
print(f"Final best SSIM (subtraction): {ssim_best_sub:.4f} (kernel: {ssim_kernel_sub}, sigma: {ssim_sigma_sub:.2f})")
print(f"Final best MSE (subtraction): {mse_best_sub:.4f} (kernel: {mse_kernel_sub}, sigma: {mse_sigma_sub:.2f})")
