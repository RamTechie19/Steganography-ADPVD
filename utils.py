import numpy as np

def pixel_difference(pixel1, pixel2):
    return np.abs(int(pixel1) - int(pixel2)) if np.isscalar(pixel1) else np.abs(pixel1.astype(int) - pixel2.astype(int))

def get_embedding_capacity(diff):
    if np.isscalar(diff):
        diff = diff
    else:
        diff = diff[0]  # Use the first channel for determining capacity
    
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
    diff = pixel_difference(pixel1, pixel2)
    capacity = get_embedding_capacity(diff)

    # Pad bits with zeros if necessary
    bits = bits.ljust(capacity, '0')

    # Convert bits to integer
    value = int(bits, 2)

    # Adjust pixel values
    if np.isscalar(pixel1):
        if pixel1 >= pixel2:
            new_pixel1 = pixel1 + value // 2
            new_pixel2 = pixel2 - (value + 1) // 2
        else:
            new_pixel1 = pixel1 - (value + 1) // 2
            new_pixel2 = pixel2 + value // 2
    else:
        if pixel1[0] >= pixel2[0]:
            new_pixel1 = pixel1 + value // 2
            new_pixel2 = pixel2 - (value + 1) // 2
        else:
            new_pixel1 = pixel1 - (value + 1) // 2
            new_pixel2 = pixel2 + value // 2

    # Ensure pixel values are within valid range
    new_pixel1 = np.clip(new_pixel1, 0, 255)
    new_pixel2 = np.clip(new_pixel2, 0, 255)

    return new_pixel1, new_pixel2

def extract_bits(pixel1, pixel2, capacity):
    if np.isscalar(pixel1):
        value = abs(int(pixel1) - int(pixel2))
    else:
        value = abs(int(pixel1[0]) - int(pixel2[0]))
    bits = format(value, f'0{capacity}b')
    return bits

