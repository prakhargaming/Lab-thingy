import cv2
import numpy as np
from skimage.metrics import normalized_root_mse as ssim
from scipy.optimize import minimize

def process_image(img, params):
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    unsharp_mask = cv2.addWeighted(img, 1 + params[0], blurred, -params[0], 0)
    
    clahe = cv2.createCLAHE(clipLimit=params[1], tileGridSize=(8, 8))
    enhanced = clahe.apply(unsharp_mask)
    
    kernel = np.array([[-1,-1,-1], [-1,9+params[2],-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    final = cv2.convertScaleAbs(sharpened, alpha=params[3], beta=params[4])
    return final

def objective_function(params, img, target):
    processed = process_image(img, params)
    return -ssim(processed, target)  # Negative because we want to maximize similarity

# Load images
img = cv2.imread('amgojs.png', cv2.IMREAD_GRAYSCALE)
target = cv2.imread('good.jpg', cv2.IMREAD_GRAYSCALE)

# Initial parameter guesses
initial_params = [0.5, 2.0, 0, 1.2, 10]

# Bounds for parameters
bounds = [(0, 2), (0, 5), (-1, 1), (0.5, 2), (-50, 50)]

# Optimize
result = minimize(objective_function, initial_params, args=(img, target), 
                  method='trust-exact', bounds=bounds)

# Get optimal parameters
optimal_params = result.x

print("Optimal parameters:", optimal_params)

# Process image with optimal parameters
final_image = process_image(img, optimal_params)

# Save and display result
cv2.imwrite('optimized_image.png', final_image)
cv2.imshow('Original', img)
cv2.imshow('Target', target)
cv2.imshow('Optimized', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()