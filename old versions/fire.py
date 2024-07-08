# %% [markdown]
# # Notebook le Epic

# %% [markdown]
# ## Imports

# %%
import numpy as np
import cv2 as cv
from PIL import Image, ImageEnhance
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

# %% [markdown]
# ## Helper Functions

# %% [markdown]
# ### imageLoad2Gray
# coverts an image into grayscale and normalizes it

# %%
def imageLoad2Gray(img: np.array, dtype = np.float32) -> np.array: 
    image = np.asarray(img, dtype = dtype)
    image = rgb2gray(image)
    image /= 255.0
    return image

# %% [markdown]
# ### img_save
# show and save a ndarray as an image

# %%
def img_save(img_array, file_name, title= '', show = True, cmap=None, dtype = np.uint8):
    
    if show:
        plt.imshow(img_array.astype(dtype), cmap = cmap)
        plt.title(title)
        plt.show()
    
    plt.imsave(file_name, img_array.astype(dtype), cmap = cmap)

# %% [markdown]
# ### brightness_pil
# 
# brightness of an image using Pillow

# %%
def brightness_pil(img, factor=0.5):
    # Convert numpy array to PIL Image
    img_pil = Image.fromarray(np.uint8(img * 255))

    # Initialize the enhancer
    enhancer = ImageEnhance.Brightness(img_pil)
    
    # Apply the enhancement
    img_output_pil = enhancer.enhance(factor)
    
    # Convert enhanced PIL image back to numpy array
    img_output = np.array(img_output_pil) / 255.0  # Normalize to [0, 1] range

    return img_output

# %% [markdown]
# ## Step 1
# 
# Convert the image into greyscale

# %%
prostate = Image.open("Prostate.jpg")
gray_prostate = imageLoad2Gray(prostate)

# %%
plt.imshow(gray_prostate, cmap="gray")

# %% [markdown]
# ## Step 2
# 
# Play around with gaussian blur

# %%
blurry_prostate = gaussian_filter(gray_prostate, sigma=15)

# %%
plt.imshow(gray_prostate/blurry_prostate, cmap="gray")
plt.imsave(arr=gray_prostate/blurry_prostate, fname= "amgojs.png", cmap="gray")

# %% [markdown]
# ## Step 3
# 
# Adjust brightness

# %%
# Read the input image
img = cv.imread('amgojs.png', cv.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
blurred = cv.GaussianBlur(img, (3, 3), 0)

# Apply unsharp masking for edge enhancement
unsharp_mask = cv.addWeighted(img, 1.5, blurred, -0.5, 0)

# Apply adaptive histogram equalization for contrast enhancement
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(unsharp_mask)

# Apply additional sharpening
kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpened = cv.filter2D(enhanced, -1, kernel)

# Adjust brightness and contrast
alpha = 1.2  # Contrast control
beta = 10    # Brightness control
final = cv.convertScaleAbs(sharpened, alpha=alpha, beta=beta)

# Save the result
cv.imwrite('enhanced_image.png', final)

# Display the original and enhanced images (optional)
cv.imshow('Original', img)
cv.imshow('Enhanced', final)
cv.waitKey(0)
cv.destroyAllWindows()