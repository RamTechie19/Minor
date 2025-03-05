#Decode_image.py
import numpy as np
from PIL import Image
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    extract_bits
)

def decode_image(stego_image, debug=False):
    try:
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        hog_features = compute_HOG(stego_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        print(f"Decoding POI indices: {poi_indices[:10]}")
        
        extracted_bits = ""
        terminator_found = False
        extracted_locations = []
        
        decoding_display_count = 0
        
        for idx, poi in enumerate(poi_indices):
            row, col = divmod(poi, width)
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col]
            pixel2 = stego_array[row, col + 1]
            
            if debug:
                print(f"Extracting at ({row},{col})")
                print(f"From pixels: {pixel1}, {pixel2}")
            
            bits = extract_bits(pixel1, pixel2, debug)
            if decoding_display_count< 20:
                print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {len(bits)}")
                print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Extracted: {bits}")
                decoding_display_count += 1
                
            extracted_bits += bits
            
            extracted_locations.append((row, col))
            
            if len(extracted_bits) >= 8:
                for i in range(0, len(extracted_bits) - 7, 8):
                    chunk = extracted_bits[i:i+8]
                    if chunk == "00000000":
                        extracted_bits = extracted_bits[:i]
                        terminator_found = True
                        break
                
                if terminator_found:
                    break

        print(f"Total pixels used for extraction: {len(extracted_locations)}")
        print(f"Total bits extracted: {len(extracted_bits)}")
        
        print(f"Binary message: {extracted_bits}")
        
        message = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                try:
                    char_bits = extracted_bits[i:i+8]
                    char_val = int(char_bits, 2)
                    if 32 <= char_val<= 126:  
                        message += chr(char_val)
                    elif debug:
                        print(f"Skipping non-printable character: {char_val} from bits {char_bits}")
                except ValueError as e:
                    if debug:
                        print(f"Error converting bits to character: {e}")
                    continue
        
        return message

    except Exception as e:
        raise Exception(f"Decoding failed: {str(e)}")
