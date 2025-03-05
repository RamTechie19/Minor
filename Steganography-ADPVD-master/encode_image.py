import numpy as np
from PIL import Image
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    embed_bits
)

def encode_image(cover_image, message, debug=False):
    try:
        cover_array = np.array(cover_image.convert("RGB"))
        height, width, _ = cover_array.shape

        binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
        print(f"Binary message: {binary_message}")
        
        bit_position = 0
        total_bits = len(binary_message)

        hog_features = compute_HOG(cover_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No suitable points of interest found in the image")

        # New: Check if there are enough POIs to embed the message
        bits_per_pair = 6  # Maximum 6 bits per pair (2 bits per channel)
        required_pairs = math.ceil(total_bits / bits_per_pair)
        if len(poi_indices) <required_pairs:
            raise ValueError(
                f"Insufficient POIs to embed the message. Required: {required_pairs}, Available: {len(poi_indices)}"
            )

        print(f"Encoding POI indices: {poi_indices[:10]}")
        
        stego_array = np.copy(cover_array)
        
        encoding_display_count = 0
        
        embedded_locations = []
        for idx, poi in enumerate(poi_indices):
            if bit_position>= total_bits:
                break
                
            row, col = divmod(poi, width)
            
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col].copy()
            pixel2 = stego_array[row, col + 1].copy()
            
            remaining_bits = total_bits - bit_position
            bits_count = min(6, remaining_bits)
            bits_to_embed = binary_message[bit_position:bit_position+bits_count]
            
            if encoding_display_count< 20:
                print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {bits_count}")
                encoding_display_count += 1
            
            if debug:
                print(f"At ({row},{col}): Embedding {bits_count} bits: {bits_to_embed}")
                print(f"Original pixels: {pixel1}, {pixel2}")
            
            new_pixel1, new_pixel2 = embed_bits(pixel1, pixel2, bits_to_embed, debug)
            
            if debug:
                print(f"Modified pixels: {new_pixel1}, {new_pixel2}")
            
            stego_array[row, col] = new_pixel1
            stego_array[row, col + 1] = new_pixel2
            
            embedded_locations.append((row, col))
            bit_position += bits_count
            
            if encoding_display_count<= 20:
                print(f"Encoding at ({row},{col}): {pixel1} → {new_pixel1}, {pixel2} → {new_pixel2} | Embedded: {bits_to_embed}")
        
        print(f"Total pixels used for embedding: {len(embedded_locations)}")
        print(f"Total bits embedded: {bit_position}")
            
        if bit_position<total_bits:
            raise ValueError(f"Image capacity insufficient. Embedded {bit_position}/{total_bits} bits")
        if debug:
            print(f"Message embedded successfully")
            print(f"First 10 embedding locations: {embedded_locations[:10]}")
        return Image.fromarray(stego_array.astype('uint8'))
    except Exception as e:
        raise Exception(f"Encoding failed: {str(e)}")
