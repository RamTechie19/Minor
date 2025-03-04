import numpy as np
import cv2

# Histogram of Oriented Gradients (HOG) to identify texture-rich regions
def compute_HOG(image_array, debug=False):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(gray).flatten()
    hog_features = (hog_features - np.min(hog_features)) / (np.max(hog_features) - np.min(hog_features))
    
    if debug:
        print(f"HOG Features shape: {hog_features.shape}")
        print(f"HOG Features range: {np.min(hog_features)} to {np.max(hog_features)}")
        print(f"HOG Features mean: {np.mean(hog_features)}")
        print(f"HOG Features first 10 values: {hog_features[:10]}")
    
    return hog_features

# Adaptive threshold for POI selection
def compute_threshold(hog_features, debug=False):
    threshold = np.mean(hog_features) * 0.8
    
    if debug:
        print(f"Computed threshold: {threshold}")
        print(f"HOG mean: {np.mean(hog_features)}")
        print(f"HOG values above threshold: {np.sum(hog_features > threshold)}/{len(hog_features)}")
    
    return threshold

# Points of Interest (POI) where data can be hidden effectively
def identify_POI(hog_features, threshold, debug=False):
    poi_indices = np.where(hog_features > threshold)[0]
    selected_indices = poi_indices[::2]  # Take every other POI
    
    if debug:
        print(f"Total POI found: {len(poi_indices)}")
        print(f"Selected POI: {len(selected_indices)}")
        print(f"First 10 POI values: {[hog_features[i] for i in selected_indices[:10]]}")
    
    return selected_indices

# Simple LSB embedding in all three channels
def embed_bits(pixel1, pixel2, bits_to_embed, debug=False):
    """
    Embed bits into the LSB of all 3 channels of both pixels.
    This provides up to 6 bits of capacity per pixel pair.
    """
    new_pixel1 = pixel1.copy()
    new_pixel2 = pixel2.copy()
    
    if len(bits_to_embed) == 0:
        return new_pixel1, new_pixel2
    
    # Track bits embedded
    bits_embedded = 0
    
    # Embed in all three RGB channels, one bit per channel
    for channel in range(3):
        if bits_embedded < len(bits_to_embed):
            # Modify LSB of pixel1's current channel
            new_pixel1[channel] = (pixel1[channel] & 0xFE) | int(bits_to_embed[bits_embedded])
            bits_embedded += 1
            
            if debug:
                print(f"Embed in P1[{channel}]: {bits_to_embed[bits_embedded-1]}, " 
                      f"Before: {pixel1[channel]}, After: {new_pixel1[channel]}")
        
        if bits_embedded < len(bits_to_embed):
            # Modify LSB of pixel2's current channel
            new_pixel2[channel] = (pixel2[channel] & 0xFE) | int(bits_to_embed[bits_embedded])
            bits_embedded += 1
            
            if debug:
                print(f"Embed in P2[{channel}]: {bits_to_embed[bits_embedded-1]}, "
                    f"Before: {pixel2[channel]}, After: {new_pixel2[channel]}")
    
    return new_pixel1, new_pixel2

# Extract bits from all three channels
def extract_bits(pixel1, pixel2, debug=False):
    """
    Extract bits from the LSB of all 3 channels of both pixels.
    This extracts up to 6 bits per pixel pair.
    """
    extracted = ""
    
    # Extract from all three RGB channels
    for channel in range(3):
        # Extract LSB from pixel1's current channel
        bit1 = pixel1[channel] & 0x01
        extracted += str(bit1)
        
        if debug:
            print(f"Extract from P1[{channel}]: {bit1}")
        
        # Extract LSB from pixel2's current channel
        bit2 = pixel2[channel] & 0x01
        extracted += str(bit2)
        
        if debug:
            print(f"Extract from P2[{channel}]: {bit2}")
    
    return extracted

# Import the encode_image and decode_image functions from adpvd.py
from adpvd import encode_image_adpvd as encode_image
from adpvd import decode_image_adpvd as decode_image