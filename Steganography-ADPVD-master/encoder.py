import numpy as np
from PIL import Image
from utils import (
    compute_HOG, compute_threshold, identify_POI,
    embed_bits
)

def encode_image(cover_image, message, debug=False):
    try:
        # Converting image to numpy array
        cover_array = np.array(cover_image.convert("RGB"))
        height, width, _ = cover_array.shape

        # Converting message to binary with null terminator
        binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
        print(f"Binary message: {binary_message}")
        
        # Track current position in binary message
        bit_position = 0
        total_bits = len(binary_message)

        # Compute HOG and identify POI
        hog_features = compute_HOG(cover_array)
        threshold = compute_threshold(hog_features)
        poi_indices = identify_POI(hog_features, threshold)

        if len(poi_indices) == 0:
            raise ValueError("No suitable points of interest found in the image")

        print(f"Encoding POI indices: {poi_indices[:10]}")
        
        # Creating a copy for embedding
        stego_array = np.copy(cover_array)
        
        # Embedding the message bits
        embedded_locations = []
        for idx, poi in enumerate(poi_indices):
            if bit_position >= total_bits:
                break
                
            row, col = divmod(poi, width)
            
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col].copy()
            pixel2 = stego_array[row, col + 1].copy()
            
            bits_to_embed = binary_message[bit_position:bit_position+2]
            if not bits_to_embed:  
                break
                
            print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {len(bits_to_embed)}")
            
            # Embed bits
            new_pixel1, new_pixel2 = embed_bits(pixel1, pixel2, bits_to_embed)
            
            if debug:
                print(f"Modified pixels: {new_pixel1}, {new_pixel2}")
            
            
            stego_array[row, col] = new_pixel1
            stego_array[row, col + 1] = new_pixel2
            
            # Track embedding location
            embedded_locations.append((row, col))
            
            # Update bit position
            bit_position += len(bits_to_embed)
            
        if bit_position < total_bits:
            raise ValueError(f"Image capacity insufficient. Embedded {bit_position}/{total_bits} bits")

        if debug:
            print(f"Message embedded successfully")
            print(f"Used {len(embedded_locations)} pixel pairs")
            print(f"First 10 embedding locations: {embedded_locations[:10]}")
            
        return Image.fromarray(stego_array.astype('uint8'))

    except Exception as e:
        raise Exception(f"Encoding failed: {str(e)}")