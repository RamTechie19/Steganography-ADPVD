from PIL import Image
import numpy as np
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    pixel_difference, get_embedding_capacity, embed_bits
)

def encode_image(cover_image, message):
    """Encode a message into an image using P-ADPVD technique."""
    try:
        # Convert image to numpy array
        cover_array = np.array(cover_image.convert("RGB"))
        height, width, _ = cover_array.shape

        # Convert message to binary with null terminator
        binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
        print(f"Binary message: {binary_message}")

        # Compute HOG and identify POI
        hog_features = compute_HOG(cover_array)
        threshold = compute_threshold(hog_features)
        poi_indices = identify_POI(hog_features, threshold)

        if len(poi_indices) == 0:
            raise ValueError("No suitable points of interest found in the image")

        print(f"Encoding POI indices: {poi_indices[:10]}")

        # Embed message
        message_index = 0
        for i in poi_indices:
            if message_index >= len(binary_message):
                break

            row, col = divmod(i, width)
            pixel1, pixel2 = cover_array[row, col], cover_array[row, (col+1) % width]

            diff = pixel_difference(pixel1, pixel2)
            capacity = get_embedding_capacity(diff)

            print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {capacity}")

            bits_to_embed = binary_message[message_index:message_index + capacity]
            new_pixel1, new_pixel2 = embed_bits(pixel1, pixel2, bits_to_embed)

            cover_array[row, col] = new_pixel1
            cover_array[row, (col+1) % width] = new_pixel2
            message_index += capacity

        if message_index < len(binary_message):
            raise ValueError("Image capacity insufficient for message length")

        return Image.fromarray(cover_array.astype('uint8'))

    except Exception as e:
        raise Exception(f"Encoding failed: {str(e)}")