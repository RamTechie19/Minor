import numpy as np
import cv2

# Histogram of Oriented Gradients (HOG) to identify texture-rich regions
def compute_HOG(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray).flatten()
    hog_features = (hog_features - np.min(hog_features)) / (np.max(hog_features) - np.min(hog_features))
    return hog_features

# adaptive threshold for POI selection
def compute_threshold(hog_features):
    return np.mean(hog_features) * 0.8

# Points of Interest (POI) where data can be hidden effectively
def identify_POI(hog_features, threshold):
    poi_indices = np.where(hog_features > threshold)[0]
    return poi_indices[::2]

# difference between two pixel values 
def pixel_difference(pixel1, pixel2):
    return int(pixel1[0]) - int(pixel2[0])

# bits can be hidden
def get_embedding_capacity(diff):
    abs_diff = abs(diff)
    if abs_diff < 8:
        return 2
    elif abs_diff < 16:
        return 3
    elif abs_diff < 32:
        return 4
    elif abs_diff < 64:
        return 5
    else:
        return 6

# Embedding bits
def embed_bits(pixel1, pixel2, bits):
    if len(bits) == 0:
        return pixel1.copy(), pixel2.copy()
        
    new_pixel1 = pixel1.copy()
    new_pixel2 = pixel2.copy()
    
    if len(bits) > 0:
        new_pixel1[0] = (new_pixel1[0] & 0xFE) | int(bits[0])
            
    if len(bits) > 1:
        new_pixel2[0] = (new_pixel2[0] & 0xFE) | int(bits[1])
            
    return new_pixel1, new_pixel2

# Extracting bits
def extract_bits(pixel1, pixel2):
    extracted = []
    
    bit1 = pixel1[0] & 0x01
    extracted.append(str(bit1))
    
    bit2 = pixel2[0] & 0x01
    extracted.append(str(bit2))
    
    return ''.join(extracted)


def decode_image(stego_image):
    try:
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        hog_features = compute_HOG(stego_array)
        threshold = compute_threshold(hog_features)
        poi_indices = identify_POI(hog_features, threshold)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        extracted_bits = ""
        terminator_found = False
        
        for idx, poi in enumerate(poi_indices):
            row, col = divmod(poi, width)
            pixel1 = stego_array[row, col]
            pixel2 = stego_array[row, (col+1) % width]
            
            bits = extract_bits(pixel1, pixel2)
            extracted_bits += bits
            
            if len(extracted_bits) >= 8:
                for i in range(0, len(extracted_bits) - 7, 8):
                    chunk = extracted_bits[i:i+8]
                    if chunk == "00000000":
                        extracted_bits = extracted_bits[:i]
                        terminator_found = True
                        break
                
                if terminator_found:
                    break

        # Convert binary to text with validation
        message = ""
        for i in range(0, len(extracted_bits), 8):
            if i + 8 <= len(extracted_bits):
                try:
                    char_bits = extracted_bits[i:i+8]
                    char_val = int(char_bits, 2)
                    if 32 <= char_val <= 126:  # Printable ASCII range
                        message += chr(char_val)
                except ValueError:
                    continue

        return message

    except Exception as e:
        raise Exception(f"Decoding failed: {str(e)}")