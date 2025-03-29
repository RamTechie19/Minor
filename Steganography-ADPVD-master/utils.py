import numpy as np
import cv2

def compute_HOG(image_array, debug=False):
    """
    Compute HOG features for the image with modified parameters to reduce feature count.
    Modification: Use larger cell size (16x16 instead of 8x8) and adjust block size and stride.
    """
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    # Modified HOG parameters: larger cell size to reduce the number of features
    hog = cv2.HOGDescriptor(
        _winSize=(image_array.shape[1], image_array.shape[0]),  # 512x512
        _blockSize=(32, 32),  # 32x32 block (2x2 cells)
        _blockStride=(16, 16),  # Stride equal to cell size, reducing overlap
        _cellSize=(16, 16),  # Larger cell size to reduce granularity
        _nbins=9  # Keep default number of bins
    )
    hog_features = hog.compute(gray).flatten()
    # Normalize features to [0, 1]
    hog_features = (hog_features - np.min(hog_features)) / (np.max(hog_features) - np.min(hog_features) + 1e-10)
    
    if debug:
        print(f"HOG Features shape: {hog_features.shape}")
        print(f"HOG Features range: {np.min(hog_features)} to {np.max(hog_features)}")
        print(f"HOG Features mean: {np.mean(hog_features)}")
        print(f"HOG Features first 10 values: {hog_features[:10]}")
    
    return hog_features

def compute_threshold(hog_features, debug=False):
    """
    Compute a stricter threshold to select fewer POIs.
    Modification: Increase multiplier from 0.8 to 1.2 to be more selective.
    """
    threshold = np.mean(hog_features) * 1.2  # Stricter threshold
    
    if debug:
        print(f"Computed threshold: {threshold}")
        print(f"HOG mean: {np.mean(hog_features)}")
        print(f"HOG values above threshold: {np.sum(hog_features > threshold)}/{len(hog_features)}")
    
    return threshold

def identify_POI(hog_features, threshold, debug=False):
    """
    Identify Points of Interest (POIs) and subsample more aggressively.
    Modification: Take every 10th POI instead of every 2nd to reduce the count.
    """
    poi_indices = np.where(hog_features > threshold)[0]
    selected_indices = poi_indices[::10]  # More aggressive subsampling
    
    if debug:
        print(f"Total POI found: {len(poi_indices)}")
        print(f"Selected POI: {len(selected_indices)}")
        print(f"First 10 POI values: {[hog_features[i] for i in selected_indices[:10]]}")
    
    return selected_indices

'''def embed_bits(pixel1, pixel2, bits_to_embed, debug=False):
    """
    Embed bits into pixel pairs (unchanged).
    """
    new_pixel1 = pixel1.copy()
    new_pixel2 = pixel2.copy()
    
    if len(bits_to_embed) == 0:
        return new_pixel1, new_pixel2
    
    bits_embedded = 0
    
    for channel in range(3):
        if bits_embedded < len(bits_to_embed):
            new_pixel1[channel] = (pixel1[channel] & 0xFE) | int(bits_to_embed[bits_embedded])
            bits_embedded += 1
            
            if debug:
                print(f"Embed in P1[{channel}]: {bits_to_embed[bits_embedded-1]}, " 
                      f"Before: {pixel1[channel]}, After: {new_pixel1[channel]}")
        
        if bits_embedded < len(bits_to_embed):
            new_pixel2[channel] = (pixel2[channel] & 0xFE) | int(bits_to_embed[bits_embedded])
            bits_embedded += 1
            
            if debug:
                print(f"Embed in P2[{channel}]: {bits_to_embed[bits_embedded-1]}, "
                    f"Before: {pixel2[channel]}, After: {new_pixel2[channel]}")
    
    return new_pixel1, new_pixel2

def extract_bits(pixel1, pixel2, debug=False):
    """
    Extract bits from pixel pairs (unchanged).
    """
    extracted = ""
    
    for channel in range(3):
        bit1 = pixel1[channel] & 0x01
        extracted += str(bit1)
        
        if debug:
            print(f"Extract from P1[{channel}]: {bit1}")
        
        bit2 = pixel2[channel] & 0x01
        extracted += str(bit2)
        
        if debug:
            print(f"Extract from P2[{channel}]: {bit2}")
    
    return extracted
    '''

from adpvd import encode_image_adpvd as encode_image
from adpvd import decode_image_adpvd as decode_image
