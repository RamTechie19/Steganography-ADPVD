from PIL import Image
import numpy as np
from utils import pixel_difference, get_embedding_capacity, extract_bits

def decode_image(stego_image):
    # Convert image to numpy array
    stego_array = np.array(stego_image)

    # Check if the image is grayscale or color
    if len(stego_array.shape) == 2:
        # Grayscale image
        flat_image = stego_array.reshape(-1)
    elif len(stego_array.shape) == 3:
        # Color image
        flat_image = stego_array.reshape(-1, 3)
    else:
        raise ValueError("Unsupported image format")

    # Initialize variables
    extracted_bits = ""
    pixel_index = 0

    # Extract the message
    while True:
        if pixel_index + 1 >= len(flat_image):
            break

        # Get two consecutive pixels
        pixel1 = flat_image[pixel_index]
        pixel2 = flat_image[pixel_index + 1]

        # Calculate pixel difference and embedding capacity
        diff = pixel_difference(pixel1, pixel2)
        capacity = get_embedding_capacity(diff)

        # Extract bits from the pixels
        bits = extract_bits(pixel1, pixel2, capacity)
        extracted_bits += bits

        # Move to the next set of pixels
        pixel_index += 2

        # Check for null terminator
        if extracted_bits[-8:] == '00000000':
            extracted_bits = extracted_bits[:-8]  # Remove null terminator
            break

    # Convert binary message to text
    message = ""
    for i in range(0, len(extracted_bits), 8):
        byte = extracted_bits[i:i+8]
        if len(byte) == 8:
            message += chr(int(byte, 2))

    return message