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
        parser.add_argument("--debug", action="store_true", help="Enable detailed debugging")

        args = parser.parse_args()

        if args.action == "encode":
            if not args.message or not args.output:
                parser.error("Encoding requires --message and --output arguments")

            # Load and encoding of the message into the image
            cover_image = Image.open(args.input_image)
            stego_image = encode_image(cover_image, args.message)

            # Debug information
            if args.debug:
                cover_array = np.array(cover_image.convert("RGB"))
                stego_array = np.array(stego_image.convert("RGB"))
                print(f"Before saving (first 5 pixels): {cover_array[:1, :5]}")
                print(f"After encoding (first 5 pixels): {stego_array[:1, :5]}")
                
                # Calculating and printing the pixel value differences 
                diff_count = np.sum(cover_array != stego_array)
                print(f"Total pixel value changes: {diff_count}")
                
                # Compare original message with decoding
                test_decoded = decode_image(stego_image)
                print(f"Test decode before saving: '{test_decoded}'")
                print(f"Original message: '{args.message}'")
                if test_decoded == args.message:
                    print("✓ Messages match before saving")
                else:
                    print("✗ Messages do not match before saving")

            # Saving the image
            stego_image.save(args.output, format='PNG')
            saved_image = Image.open(args.output)
            
            # Verifying the saved image
            if args.debug:
                saved_array = np.array(saved_image.convert("RGB"))
                print(f"After saving (first 5 pixels): {saved_array[:1, :5]}")
                
                if np.array_equal(stego_array, saved_array):
                    print("✓ Saved image matches stego image")
                else:
                    print("✗ Saved image differs from stego image")
                    diff_count = np.sum(stego_array != saved_array)
                    print(f"Saving introduced {diff_count} differences")
                
                # Test decode after saving
                saved_decoded = decode_image(saved_image)
                print(f"Test decode after saving: '{saved_decoded}'")
                if saved_decoded == args.message:
                    print("✓ Messages match after saving")
                else:
                    print("✗ Messages do not match after saving")

            print(f"Message encoded successfully. Stego image saved as {args.output}")

        elif args.action == "decode":
            
            # Load and decoding of the image for message 
            stego_image = Image.open(args.input_image)
            message = decode_image(stego_image)
            print(f"Decoded message: '{message}'")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())