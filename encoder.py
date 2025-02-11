from PIL import Image
import numpy as np
from utils import pixel_difference, get_embedding_capacity, embed_bits

def encode_image(cover_image, message):
    # Get the original image mode
    original_mode = cover_image.mode

    # Convert image to RGB if it's not already
    if original_mode != 'RGB':
        cover_image = cover_image.convert('RGB')

    # Convert image to numpy array
    cover_array = np.array(cover_image)
    height, width, channels = cover_array.shape

    # Convert message to binary
    binary_message = ''.join(format(ord(char), '08b') for char in message)
    binary_message += '00000000'  # Add null terminator

    # Flatten the image array
    flat_image = cover_array.reshape(-1, channels)

    # Initialize variables
    message_index = 0
    pixel_index = 0

    # Embed the message
    while message_index < len(binary_message):
        if pixel_index + 1 >= len(flat_image):
            raise ValueError("Message is too long for this image")

        # Get two consecutive pixels
        pixel1 = flat_image[pixel_index]
        pixel2 = flat_image[pixel_index + 1]

        # Calculate pixel difference and embedding capacity
        diff = pixel_difference(pixel1, pixel2)
        capacity = get_embedding_capacity(diff)

        # Embed bits in the pixels
        bits_to_embed = binary_message[message_index:message_index + capacity]
        new_pixel1, new_pixel2 = embed_bits(pixel1, pixel2, bits_to_embed)

        # Update the image array
        flat_image[pixel_index] = new_pixel1
        flat_image[pixel_index + 1] = new_pixel2

        # Move to the next set of pixels and bits
        pixel_index += 2
        message_index += capacity

    # Reshape the flattened array back to the original image shape
    stego_array = flat_image.reshape(height, width, channels)

    # Create a new image from the modified array
    stego_image = Image.fromarray(stego_array.astype('uint8'), 'RGB')

    # Convert back to the original mode if necessary
    if original_mode != 'RGB':
        stego_image = stego_image.convert(original_mode)

    return stego_image

