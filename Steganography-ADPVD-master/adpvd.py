import numpy as np
from PIL import Image
import math
from utils import compute_HOG, compute_threshold, identify_POI
from tabulate import tabulate

# Range tables for ADPVD (unchanged)
def get_range_table():
    return [
        (0, 7, 3),    
        (8, 15, 3),      
        (16, 31, 4),      
        (32, 63, 5),     
        (64, 127, 6),    
        (128, 255, 7)    
    ]

def find_range(diff, range_table):
    for r in range_table:
        if r[0] <= diff <= r[1]:
            return r
    return range_table[-1]  

def compute_new_difference(diff, data_bits, range_info):
    lower, upper, bit_depth = range_info
    data_value = int(data_bits, 2)
    new_diff = lower + data_value
    return new_diff

def adjust_pixel_values(p1, p2, new_diff):
    old_diff = abs(int(p2) - int(p1))
    diff_diff = new_diff - old_diff
    m = math.ceil(diff_diff / 2)
    n = math.floor(diff_diff / 2)
    if p1 <= p2:
        new_p1 = max(0, min(255, int(p1) - m))
        new_p2 = max(0, min(255, int(p2) + n))
    else:
        new_p1 = max(0, min(255, int(p1) + m))
        new_p2 = max(0, min(255, int(p2) - n))
        
    actual_diff = abs(new_p2 - new_p1)
    if actual_diff != new_diff:
        if p1 <= p2:
            new_p2 = max(0, min(255, new_p1 + new_diff))
        else:
            new_p2 = max(0, min(255, new_p1 - new_diff))
    
    return int(new_p1), int(new_p2)

def extract_from_difference(diff, range_info):
    lower, upper, bit_depth = range_info
    data_value = diff - lower
    data_bits = format(data_value, f'0{bit_depth}b')
    return data_bits

def encode_image_adpvd(cover_image, message, debug=False):
    try:
        cover_array = np.array(cover_image.convert("RGB"))
        height, width, _ = cover_array.shape

        binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
        print(f"Binary message: {binary_message}")
        
        bit_position = 0
        total_bits = len(binary_message)

        range_table = get_range_table()

        hog_features = compute_HOG(cover_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No suitable points of interest found in the image")

        # New: Check if there are enough POIs to embed the message
        # Assume an average of 4.5 bits per channel, 3 channels per pixel pair
        bits_per_pair = 4.5 * 3  # Approximately 13.5 bits per pair
        required_pairs = math.ceil(total_bits / bits_per_pair)
        if len(poi_indices) <required_pairs:
            raise ValueError(
                f"Insufficient POIs to embed the message. Required: {required_pairs}, Available: {len(poi_indices)}"
            )

        print(f"Encoding POI indices: {poi_indices[:10]}")
        
        stego_array = np.copy(cover_array)
        
        embedded_locations = []
        display_count = 0
        
        table_data = []
        channel_usage = {'red': 0, 'green': 0, 'blue': 0}
        bits_per_channel = {'red': 0, 'green': 0, 'blue': 0}
        
        for idx, poi in enumerate(poi_indices):
            if bit_position>= total_bits:
                break
                
            row, col = divmod(poi, width)
            
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col].copy()
            pixel2 = stego_array[row, col + 1].copy()
            
            new_pixel1 = pixel1.copy()
            new_pixel2 = pixel2.copy()
            
            if display_count< 20:
                total_capacity = 0
                for channel in range(3):
                    diff = abs(int(pixel1[channel]) - int(pixel2[channel]))
                    range_info = find_range(diff, range_table)
                    total_capacity += range_info[2]
                print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {total_capacity}")
                display_count += 1
            
            for channel in range(3):
                if bit_position>= total_bits:
                    break
                    
                diff = abs(int(pixel1[channel]) - int(pixel2[channel]))

                range_info = find_range(diff, range_table)
                lower, upper, bit_depth = range_info
                
                remaining_bits = total_bits - bit_position
                bits_to_embed = binary_message[bit_position:bit_position + min(bit_depth, remaining_bits)]
                
                if len(bits_to_embed) <bit_depth:
                    bits_to_embed = bits_to_embed.ljust(bit_depth, '0')
                    
                if debug:
                    print(f"At ({row},{col}) channel {channel}: diff={diff}, range={range_info}")
                    print(f"Embedding {len(bits_to_embed)} bits: {bits_to_embed}")
                
                new_diff = compute_new_difference(diff, bits_to_embed, range_info)
                
                new_p1, new_p2 = adjust_pixel_values(pixel1[channel], pixel2[channel], new_diff)
                
                new_pixel1[channel] = new_p1
                new_pixel2[channel] = new_p2
                
                channel_name = ['Red', 'Green', 'Blue'][channel]
                table_data.append([
                    channel_name,
                    f"({row},{col})",
                    f"{pixel1[channel]},{pixel2[channel]}",
                    diff,
                    len(bits_to_embed)
                ])
                
                channel_key = channel_name.lower()
                channel_usage[channel_key] += 1
                bits_per_channel[channel_key] += len(bits_to_embed)
                
                if debug:
                    print(f"Channel {channel}: {pixel1[channel]},{pixel2[channel]} → {new_p1},{new_p2}")
                    print(f"Diff: {diff} → {new_diff}")
                
                bit_position += bit_depth
            
            stego_array[row, col] = new_pixel1
            stego_array[row, col + 1] = new_pixel2
            
            embedded_locations.append((row, col))
            
            if display_count<= 20:
                print(f"Encoding at ({row},{col}): {pixel1} → {new_pixel1}, {pixel2} → {new_pixel2}")
        
        print(f"Total pixels used for embedding: {len(embedded_locations)}")
        print(f"Total bits embedded: {bit_position}")
            
        if bit_position<total_bits:
            raise ValueError(f"Image capacity insufficient. Embedded {bit_position}/{total_bits} bits")

        headers = ["Channel", "Pixel Position", "Pixel Values (P1,P2)", "Difference", "Bits Embedded"]
        print("\nEncoding Details Table:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print("\nChannel Usage Statistics:")
        print(f"Red Channel: {channel_usage['red']} pixels used, {bits_per_channel['red']} bits embedded")
        print(f"Green Channel: {channel_usage['green']} pixels used, {bits_per_channel['green']} bits embedded")
        print(f"Blue Channel: {channel_usage['blue']} pixels used, {bits_per_channel['blue']} bits embedded")

        return Image.fromarray(stego_array.astype('uint8'))

    except Exception as e:
        raise Exception(f"ADPVD encoding failed: {str(e)}")

def decode_image_adpvd(stego_image, debug=False):
    try:
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        range_table = get_range_table()

        hog_features = compute_HOG(stego_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        print(f"Decoding POI indices: {poi_indices[:10]}")
        
        extracted_bits = ""
        terminator_found = False
        extracted_locations = []
        display_count = 0
        
        table_data = []
        
        for idx, poi in enumerate(poi_indices):
            row, col = divmod(poi, width)
            
            if col + 1 >= width:
                continue
                
            pixel1 = stego_array[row, col]
            pixel2 = stego_array[row, col + 1]
            
            total_capacity = 0
            for channel in range(3):
                diff = abs(int(pixel1[channel]) - int(pixel2[channel]))
                range_info = find_range(diff, range_table)
                total_capacity += range_info[2]
            
            if display_count< 20:
                print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {total_capacity}")
                display_count += 1
            
            position_bits = ""
            for channel in range(3):
                diff = abs(int(pixel1[channel]) - int(pixel2[channel]))
                range_info = find_range(diff, range_table)
                bits = extract_from_difference(diff, range_info)
                position_bits += bits
                extracted_bits += bits
                
                channel_name = ['Red', 'Green', 'Blue'][channel]
                table_data.append([
                    channel_name,
                    f"({row},{col})",
                    f"{pixel1[channel]},{pixel2[channel]}",
                    diff,
                    len(bits)
                ])
                
                if debug:
                    print(f"Channel {channel}: diff={diff}, range={range_info}, extracted: {bits}")
            
            if display_count<= 20:
                print(f"Decoding at ({row},{col}): {pixel1}, {pixel2} | Extracted: {position_bits}")
            
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
        
        headers = ["Channel", "Pixel Position", "Pixel Values (P1,P2)", "Difference", "Bits Extracted"]
        print("\nDecoding Details Table:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        
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
        
        print(f"Decoded message: '{message}'")
        return message

    except Exception as e:
        raise Exception(f"ADPVD decoding failed: {str(e)}")
