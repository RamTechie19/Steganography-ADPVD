import numpy as np
import cv2

def compute_HOG(image_array):
    """Compute Histogram of Oriented Gradients (HOG) to identify texture-rich regions."""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray).flatten()
    return hog_features

def compute_threshold(hog_features):
    """Compute an adaptive threshold for POI selection."""
    return np.mean(hog_features) * 0.8

def identify_POI(hog_features, threshold):
    """Identify Points of Interest (POI) where data can be hidden effectively."""
    return np.where(hog_features > threshold)[0]

def pixel_difference(pixel1, pixel2):
    """Calculate the absolute difference between two pixel values."""
    return np.abs(int(pixel1[0]) - int(pixel2[0]))

def get_embedding_capacity(diff):
    """Determine how many bits can be hidden based on pixel difference."""
    if diff < 8:
        return 3
    elif diff < 16:
        return 4
    elif diff < 32:
        return 5
    elif diff < 64:
        return 6
    else:
        return 7

def embed_bits(pixel1, pixel2, bits):
    """Embed binary message bits into pixel pairs using adaptive P-ADPVD method."""
    diff = pixel_difference(pixel1, pixel2)
    capacity = get_embedding_capacity(diff)
    bits = bits.ljust(capacity, '0')
    value = int(bits, 2)

    if pixel1[0] >= pixel2[0]:
        new_pixel1 = np.array([pixel1[0] + value // 2] + list(pixel1[1:]))
        new_pixel2 = np.array([pixel2[0] - (value + 1) // 2] + list(pixel2[1:]))
    else:
        new_pixel1 = np.array([pixel1[0] - (value + 1) // 2] + list(pixel1[1:]))
        new_pixel2 = np.array([pixel2[0] + value // 2] + list(pixel2[1:]))

    return np.clip(new_pixel1, 0, 255), np.clip(new_pixel2, 0, 255)

def extract_bits(pixel1, pixel2):
    """Extract hidden bits from pixel pairs."""
    diff = pixel_difference(pixel1, pixel2)
    capacity = get_embedding_capacity(diff)
    value = abs(int(pixel1[0]) - int(pixel2[0]))
    return format(value, f'0{capacity}b')