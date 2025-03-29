import argparse
from PIL import Image
import numpy as np
from utils import encode_image, decode_image

def main():
    try:
        parser = argparse.ArgumentParser(description="Steganography using ADPVD technique")
        parser.add_argument("action", choices=["encode", "decode"], help="Action to perform")
        parser.add_argument("input_image", help="Path to the input image")
        parser.add_argument("--message", help="Message to encode (required for encoding)")
        parser.add_argument("--output", help="Path to save the output image (required for encoding)")
        parser.add_argument("--debug", action="store_true", help="Enable detailed debugging")

        args = parser.parse_args()

        if args.action == "encode":
            if not args.message or not args.output:
                parser.error("Encoding requires --message and --output arguments")

            cover_image = Image.open(args.input_image)
            stego_image = encode_image(cover_image, args.message, args.debug)

            if args.debug:
                cover_array = np.array(cover_image.convert("RGB"))
                stego_array = np.array(stego_image.convert("RGB"))
                print(f"Before encoding (first 5 pixels): {cover_array[:1, :5]}")
                print(f"After encoding (first 5 pixels): {stego_array[:1, :5]}")
                
                diff_count = np.sum(cover_array != stego_array)
                print(f"Total pixel value changes: {diff_count}")
                
                print("Verifying message before saving...")
                test_decoded = decode_image(stego_image, args.debug)
                print(f"Test decode before saving: '{test_decoded}'")
                print(f"Original message: '{args.message}'")
                if test_decoded == args.message:
                    print("✓ Messages match before saving")
                else:
                    print("✗ Messages do not match before saving")
                    print("\nDebug information for message mismatch:")
                    print(f"Original message length: {len(args.message)}")
                    print(f"Decoded message length: {len(test_decoded)}")
                    min_len = min(len(args.message), len(test_decoded))
                    for i in range(min_len):
                        orig = args.message[i]
                        decoded = test_decoded[i]
                        if orig != decoded:
                            print(f"Mismatch at position {i}: '{orig}' vs '{decoded}'")
                            print(f"  Original char code: {ord(orig)} ({format(ord(orig), '08b')})")
                            print(f"  Decoded char code: {ord(decoded)} ({format(ord(decoded), '08b')})")

            stego_image.save(args.output, format='TIFF')
            print(f"Message encoded successfully. Stego image saved as {args.output}")
            
            if args.debug:
                saved_image = Image.open(args.output)
                saved_array = np.array(saved_image.convert("RGB"))
                print(f"\nAfter saving (first 5 pixels): {saved_array[:1, :5]}")
                
                if np.array_equal(stego_array, saved_array):
                    print("✓ Saved image matches stego image")
                else:
                    print("✗ Saved image differs from stego image")
                    diff_count = np.sum(stego_array != saved_array)
                    print(f"Saving introduced {diff_count} differences")
                
                print("\nVerifying message after saving...")
                saved_decoded = decode_image(saved_image, args.debug)
                print(f"Test decode after saving: '{saved_decoded}'")
                if saved_decoded == args.message:
                    print("✓ Messages match after saving")
                else:
                    print("✗ Messages do not match after saving")
                    print("\nDebug information for message mismatch after saving:")
                    print(f"Original message length: {len(args.message)}")
                    print(f"Decoded message length: {len(saved_decoded)}")
                    min_len = min(len(args.message), len(saved_decoded))
                    for i in range(min_len):
                        orig = args.message[i]
                        decoded = saved_decoded[i]
                        if orig != decoded:
                            print(f"Mismatch at position {i}: '{orig}' vs '{decoded}'")
                            print(f"  Original char code: {ord(orig)} ({format(ord(orig), '08b')})")
                            print(f"  Decoded char code: {ord(decoded)} ({format(ord(decoded), '08b')})")

        elif args.action == "decode":
            stego_image = Image.open(args.input_image)
            message = decode_image(stego_image, args.debug)
            print(f"Decoded message: '{message}'")

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
