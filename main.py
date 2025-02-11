import argparse
from PIL import Image
from encoder import encode_image
from decoder import decode_image

def main():
    parser = argparse.ArgumentParser(description="Steganography using ADPVD technique")
    parser.add_argument("action", choices=["encode", "decode"], help="Action to perform")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("--message", help="Message to encode (required for encoding)")
    parser.add_argument("--output", help="Path to save the output image (required for encoding)")

    args = parser.parse_args()

    if args.action == "encode":
        if not args.message or not args.output:
            parser.error("Encoding requires --message and --output arguments")
        
        cover_image = Image.open(args.input_image)
        original_format = cover_image.format
        stego_image = encode_image(cover_image, args.message)
        
        # Save the stego image in the same format as the original image
        if original_format:
            stego_image.save(args.output, format=original_format)
        else:
            # If the original format is not available, default to PNG
            stego_image.save(args.output, format='PNG')
        
        print(f"Message encoded successfully. Stego image saved as {args.output}")

    elif args.action == "decode":
        stego_image = Image.open(args.input_image)
        message = decode_image(stego_image)
        print(f"Decoded message: {message}")

if __name__ == "__main__":
    main()



#[V0_FILE]python:file="encoder.py" type="code"
from PIL import Image
import numpy as np
from utils import pixel_difference, get_embedding_capacity, embed_bits

def encode_image(cover_image, message):
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
    stego_image = Image.fromarray(stego_array.astype('uint8'))
    return stego_image

#[V0_FILE]python:file="decoder.py" type="code"
from PIL import Image
import numpy as np
from utils import pixel_difference, get_embedding_capacity, extract_bits

def decode_image(stego_image):
    # Convert image to numpy array
    stego_array = np.array(stego_image)

    # Flatten the image array
    flat_image = stego_array.reshape(-1, 3)

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
            break

    # Convert binary message to text
    message = ""
    for i in range(0, len(extracted_bits) - 8, 8):
        byte = extracted_bits[i:i+8]
        message += chr(int(byte, 2))

    return message

#[V0_FILE]python:file="utils.py" type="code"
import numpy as np

def pixel_difference(pixel1, pixel2):
    return np.abs(pixel1.astype(int) - pixel2.astype(int))

def get_embedding_capacity(diff):
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
    diff = pixel_difference(pixel1, pixel2)
    value = abs(int(pixel1[0]) - int(pixel2[0]))
    bits = format(value, f'0{capacity}b')
    return bits
