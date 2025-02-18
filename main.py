import argparse
from PIL import Image
import numpy as np
from encoder import encode_image
from decoder import decode_image

def main():
    try:
        parser = argparse.ArgumentParser(description="Steganography using P-ADPVD technique")
        parser.add_argument("action", choices=["encode", "decode"], help="Action to perform")
        parser.add_argument("input_image", help="Path to the input image")
        parser.add_argument("--message", help="Message to encode (required for encoding)")
        parser.add_argument("--output", help="Path to save the output image (required for encoding)")

        args = parser.parse_args()

        if args.action == "encode":
            if not args.message or not args.output:
                parser.error("Encoding requires --message and --output arguments")

            # Load and encode
            cover_image = Image.open(args.input_image)
            stego_image = encode_image(cover_image, args.message)

            # Debug information
            cover_array = np.array(cover_image.convert("RGB"))
            stego_array = np.array(stego_image.convert("RGB"))
            print(f"Before saving (first 5 pixels): {cover_array[:1, :5]}")
            print(f"After encoding (first 5 pixels): {stego_array[:1, :5]}")

            # Save and verify
            stego_image.save(args.output, format='PNG')
            saved_image = Image.open(args.output)
            saved_array = np.array(saved_image.convert("RGB"))
            print(f"After saving (first 5 pixels): {saved_array[:1, :5]}")

            print(f"Message encoded successfully. Stego image saved as {args.output}")

        elif args.action == "decode":
            # Load and decode
            stego_image = Image.open(args.input_image)
            message = decode_image(stego_image)
            print(f"Decoded message: {message}")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())