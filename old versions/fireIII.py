import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.optimize import minimize
from skimage.exposure import match_histograms

def process_image(img, params):

    # blur, clahe, and sharpen
    
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    
    clahe = cv2.createCLAHE(clipLimit=params[1], tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    kernel = np.array([[-1,-1,-1], [-1,9+params[2],-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    final = cv2.convertScaleAbs(sharpened, alpha=params[3], beta=params[4])
    
    return final

def objective_function(params, img, target):
    processed = process_image(img, params)
    similarity = ssim(processed, target)
 
    return -(similarity)

# Load images   
img = cv2.imread('amgojs.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE)

initial_params = [0.3, 1.5, 0, 1.1, 20, 0.5]

# Adjust bounds to allow for more brightness increase
bounds = [(0, 1), (0, 3), (-0.5, 0.5), (0.8, 1.5), (0, 50), (0, 2)]

result = minimize(objective_function, initial_params, args=(img, target), 
                  method='L-BFGS-B', bounds=bounds)

optimal_params = result.x
print("Optimal parameters:", optimal_params)

final_image = process_image(img, optimal_params)

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
    