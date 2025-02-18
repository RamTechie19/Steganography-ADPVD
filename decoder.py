import numpy as np
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    pixel_difference, get_embedding_capacity, extract_bits
)

def decode_image(stego_image):
    """Decode a message from a stego image using P-ADPVD technique."""
    try:
        # Convert image to numpy array
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        # Compute HOG and identify POI
        hog_features = compute_HOG(stego_array)
        threshold = compute_threshold(hog_features)
        poi_indices = identify_POI(hog_features, threshold)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        print(f"Decoding POI indices: {poi_indices[:10]}")

        # Extract message
        extracted_bits = ""
        for i in poi_indices:
            row, col = divmod(i, width)
            pixel1, pixel2 = stego_array[row, col], stego_array[row, (col+1) % width]

            diff = pixel_difference(pixel1, pixel2)
            capacity = get_embedding_capacity(diff)

            print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {capacity}")

            extracted_bits += extract_bits(pixel1, pixel2)

            # Check for null terminator
            if len(extracted_bits) >= 8 and "00000000" in extracted_bits:
                extracted_bits = extracted_bits[:extracted_bits.index("00000000")]
                break

        # Convert binary to text
        message = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                char = chr(int(extracted_bits[i:i+8], 2))
                message += char

        return message

    except Exception as e:
        raise Exception(f"Decoding failed: {str(e)}")