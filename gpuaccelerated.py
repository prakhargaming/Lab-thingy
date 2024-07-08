# %% [markdown]
# ## Imports

# %%
import cupy as cp
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt

# %% [markdown]
# ## Parameters

# %%
number_of_kernel_elements = 9
kernel_size = (3,3)

# %% [markdown]
# ## Helper Functions

# %%
def process_image(img, kernel_params, kernel_size = (3, 3)):
    # Reshape 9-element kernel_params into a 3x3 matrix
    kernel = np.array(kernel_params).reshape(kernel_size)
    
    # Apply convolution using OpenCV with CUDA
    kernel_gpu = cv2.cuda_GpuMat()
    kernel_gpu.upload(kernel)
    
    img_gpu = cv2.cuda_GpuMat()
    img_gpu.upload(img)
    
    processed_gpu = cv2.cuda.filter2D(img_gpu, -1, kernel_gpu)
    processed = processed_gpu.download()
    
    return processed

# %%
def objective_function(kernel_params, img, target):
    processed = process_image(img, kernel_params)
    similarity = ssim(processed, target)
    
    return -similarity  # Minimize negative SSIM for optimization

# %%
def show_image(img, cmap = None):
    %matplotlib inline
    plt.imshow(img, cmap)
    plt.show()

# %%
def save_img_arr(imgs, fname = "img array"):
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in imgs:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save('test.jpg')

# %% [markdown]
# ## Algorithm

# %%
# Load images
img = cv2.imread('Prostate.jpg', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE)

# %%
show_image(img, cmap="gray")

# %%
# Initial kernel parameters (random initialization)
initial_kernel_params = np.random.uniform(0, 1, size=number_of_kernel_elements)

# Bounds for kernel parameter optimization (adjust as needed)
kernel_bounds = [(0, 1)] * number_of_kernel_elements

# Optimization using a different method (e.g., SLSQP)
result = minimize(objective_function, initial_kernel_params, args=(img, target),
                  method='Nelder-Mead', bounds=kernel_bounds)

optimal_kernel_params = result.x
print("Optimal kernel parameters:")
print(optimal_kernel_params)

# Reshape optimal parameters into a 3x3 kernel matrix
optimal_kernel_matrix = optimal_kernel_params.reshape(kernel_size)

# Process input image with optimal kernel
final_image = process_image(img, optimal_kernel_params)

# Histogram matching
final_image = match_histograms(final_image, target)

# Additional brightness correction
mean_target = np.mean(target)
mean_final = np.mean(final_image)
brightness_diff = mean_target - mean_final
final_image = cv2.add(final_image, np.full(final_image.shape, brightness_diff, dtype=final_image.dtype))

# Ensure the image is in the correct range
final_image = np.clip(final_image, 0, 255).astype(np.uint8)

# %%
height, width = final_image.shape

# Create a blank canvas that can hold all three images side by side
combined_image = np.zeros((height, width * 3), dtype=final_image.dtype)

# Place each image onto the canvas
combined_image[:, :width] = final_image
combined_image[:, width:width*2] = target
combined_image[:, width*2:] = img

# Save the combined image
cv2.imwrite('combined_image.png', combined_image)

# Display the images using matplotlib with titles
f, axarr = plt.subplots(1, 3, figsize=(15, 5))
axarr[0].imshow(final_image, cmap="gray")
axarr[0].set_title('Final Image')
axarr[1].imshow(img, cmap="gray")
axarr[1].set_title('Original Image')
axarr[2].imshow(target, cmap="gray")
axarr[2].set_title('Target Image')

# Remove axis ticks
for ax in axarr:
    ax.axis('off')

plt.show()
