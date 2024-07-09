import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
from skimage.exposure import match_histograms

number_of_kernel_elements = 9
kernel_size = (3,3)

def process_image(img, kernel_params, kernel_size = (3, 3)):
    # Reshape 9-element kernel_params into a 3x3 matrix
    kernel = np.array(kernel_params).reshape(kernel_size)
    
    # Apply convolution
    processed = cv2.filter2D(img, -1, kernel)
    
    return processed

def objective_function(kernel_params, img, target):
    processed = process_image(img, kernel_params)
    similarity = ssim(processed, target)
    
    return -similarity  # Minimize negative SSIM for optimization

# Load images
img = cv2.imread('prostate.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE)

# Initial kernel parameters (random initialization)
initial_kernel_params = np.random.uniform(0, 1, size=number_of_kernel_elements)

# Bounds for kernel parameter optimization (adjust as needed)
kernel_bounds = [(0, 1)] * number_of_kernel_elements

# Optimization using a different method (e.g., SLSQP)
result = minimize(objective_function, initial_kernel_params, args=(img, target),
                  method='SLSQP', bounds=kernel_bounds)

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

cv2.imwrite('optimized_image.png', final_image)
cv2.imshow('Original', img)
cv2.imshow('Target', target)
cv2.imshow('Optimized', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
