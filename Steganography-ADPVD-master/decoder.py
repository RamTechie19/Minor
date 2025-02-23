import numpy as np
from PIL import Image
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    extract_bits
)

def decode_image(stego_image, debug=False):
    """
    Decode a message from a stego image using LSB extraction at HOG-identified positions.
    
    Args:
        stego_image (PIL.Image): The stego image containing the hidden message
        debug (bool): Whether to print debug information
        
    Returns:
        str: The decoded message
    """
    try:
        # Convert image to numpy array
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        # Compute HOG and identify POI - must match encoding process
        hog_features = compute_HOG(stego_array)
        threshold = compute_threshold(hog_features)
        poi_indices = identify_POI(hog_features, threshold)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        print(f"Decoding POI indices: {poi_indices[:10]}")
        
        # Extract bits
        extracted_bits = ""
        terminator_found = False
        extracted_locations = []
        
        for idx, poi in enumerate(poi_indices):
            row, col = divmod(poi, width)
            
            # Ensure we don't wrap around to next row
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col]
            pixel2 = stego_array[row, col + 1]
            
            if debug:
                print(f"Extracting at ({row},{col})")
                print(f"From pixels: {pixel1}, {pixel2}")
            
            # Extract 2 bits from this POI
            bits = extract_bits(pixel1, pixel2)
            extracted_bits += bits
            
            print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {len(bits)}")
            
            # Track extraction location
            extracted_locations.append((row, col))
            
            if debug:
                print(f"Extracted bits: {bits}")
            
            # Check for null terminator every 8 bits
            if len(extracted_bits) >= 8:
                for i in range(0, len(extracted_bits) - 7, 8):
                    chunk = extracted_bits[i:i+8]
                    if chunk == "00000000":
                        extracted_bits = extracted_bits[:i]
                        terminator_found = True
                        break
                
                if terminator_found:
                    break

        
        if debug:
            print(f"Total extracted bits: {len(extracted_bits)}")
            print(f"First 10 extraction locations: {extracted_locations[:10]}")

        # Convert binary to text with validation
        message = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                try:
                    char_bits = extracted_bits[i:i+8]
                    char_val = int(char_bits, 2)
                    if 32 <= char_val <= 126:  # Printable ASCII range
                        message += chr(char_val)
                    elif debug:
                        print(f"Skipping non-printable character: {char_val}")
                except ValueError as e:
                    if debug:
                        print(f"Error converting bits to character: {e}")
                    continue
        print(f"Binary message: {extracted_bits}")
        if debug:
            print(f"Decoded message: '{message}'")

        return message

    except Exception as e:
        raise Exception(f"Decoding failed: {str(e)}")