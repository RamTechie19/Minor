import numpy as np
from PIL import Image
import math
from utils import compute_HOG, compute_threshold, identify_POI
from tabulate import tabulate

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

def calculate_direction_capacity(cover_array, direction, channel, height, width, range_table):
    """
    Calculate the embedding capacity for a given direction and channel using non-overlapping pixel pairs.
    direction: 'horizontal', 'vertical', or 'diagonal'
    channel: 0 (Red), 1 (Green), 2 (Blue)
    """
    capacity = 0
    if direction == 'horizontal':
        for row in range(height):
            for col in range(0, width - 1, 2):  # Step by 2 to avoid overlap
                pixel1 = cover_array[row, col, channel]
                pixel2 = cover_array[row, col + 1, channel]
                diff = abs(int(pixel1) - int(pixel2))
                range_info = find_range(diff, range_table)
                bit_depth = range_info[2]
                capacity += bit_depth
    elif direction == 'vertical':
        for col in range(width):
            for row in range(0, height - 1, 2):  # Step by 2 to avoid overlap
                pixel1 = cover_array[row, col, channel]
                pixel2 = cover_array[row + 1, col, channel]
                diff = abs(int(pixel1) - int(pixel2))
                range_info = find_range(diff, range_table)
                bit_depth = range_info[2]
                capacity += bit_depth
    elif direction == 'diagonal':
        # Pass 1: (row, col) where row % 2 == 0 and col % 2 == 0
        for row in range(0, height - 1, 2):
            for col in range(0, width - 1, 2):
                pixel1 = cover_array[row, col, channel]
                pixel2 = cover_array[row + 1, col + 1, channel]
                diff = abs(int(pixel1) - int(pixel2))
                range_info = find_range(diff, range_table)
                bit_depth = range_info[2]
                capacity += bit_depth
        # Pass 2: (row, col) where row % 2 == 1 and col % 2 == 1
        for row in range(1, height - 1, 2):
            for col in range(1, width - 1, 2):
                pixel1 = cover_array[row, col, channel]
                pixel2 = cover_array[row + 1, col + 1, channel]
                diff = abs(int(pixel1) - int(pixel2))
                range_info = find_range(diff, range_table)
                bit_depth = range_info[2]
                capacity += bit_depth
    return capacity

def encode_image_adpvd(cover_image, message, debug=False):
    try:
        cover_array = np.array(cover_image.convert("RGB"))
        height, width, _ = cover_array.shape

        # Calculate capacities using non-overlapping pairs
        range_table = get_range_table()
        directions = ['horizontal', 'vertical', 'diagonal']
        channels = ['Red', 'Green', 'Blue']
        capacities = {channel: {direction: 0 for direction in directions} for channel in channels}

        for channel_idx, channel_name in enumerate(channels):
            for direction in directions:
                capacity = calculate_direction_capacity(cover_array, direction, channel_idx, height, width, range_table)
                capacities[channel_name][direction] = capacity

        # Display capacities
        table_data = []
        for channel in channels:
            row = [channel] + [capacities[channel][direction] for direction in directions]
            table_data.append(row)
        headers = ["Channel"] + [direction.capitalize() for direction in directions]
        print("\nEmbedding Capacity Table (in bits):")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        # Determine the best direction for each channel
        best_directions = {}
        for channel in channels:
            best_direction = max(capacities[channel], key=capacities[channel].get)
            best_directions[channel] = best_direction
        print("\nBest Directions for Each Channel:")
        for channel, direction in best_directions.items():
            print(f"{channel}: {direction.capitalize()}")

        # Total capacity
        total_capacity = sum(max(capacities[channel].values()) for channel in channels)
        print(f"\nTotal Embedding Capacity: {total_capacity} bits")

        # Encoding process
        binary_message = ''.join(format(ord(c), '08b') for c in message) + '00000000'
        if debug:
            print(f"Binary message: {binary_message}")

        bit_position = 0
        total_bits = len(binary_message)

        hog_features = compute_HOG(cover_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No suitable points of interest found in the image")

        if debug:
            print(f"Encoding POI indices (first 10): {poi_indices[:10]}")

        stego_array = np.copy(cover_array)
        embedded_locations = []
        used_positions = set()
        display_count = 0
        table_data = []
        channel_usage = {'red': 0, 'green': 0, 'blue': 0}
        bits_per_channel = {'red': 0, 'green': 0, 'blue': 0}

        # Convert POI indices to a list of (row, col) pairs
        poi_positions = [(divmod(poi, width)) for poi in poi_indices]

        # Process diagonal pairs in two passes if needed
        for channel_idx, channel_name in enumerate(channels):
            if bit_position >= total_bits:
                break

            direction = best_directions[channel_name]
            if direction != 'diagonal':
                # Handle horizontal and vertical as before
                for row, col in poi_positions:
                    if bit_position >= total_bits:
                        break

                    pixel1, pixel2 = None, None
                    position_key = None

                    if direction == 'horizontal' and col + 1 < width:
                        if col % 2 == 0:
                            position_key = (row, col, direction)
                            if position_key in used_positions:
                                continue
                            pixel1 = stego_array[row, col].copy()
                            pixel2 = stego_array[row, col + 1].copy()
                            used_positions.add(position_key)
                        else:
                            continue
                    elif direction == 'vertical' and row + 1 < height:
                        if row % 2 == 0:
                            position_key = (row, col, direction)
                            if position_key in used_positions:
                                continue
                            pixel1 = stego_array[row, col].copy()
                            pixel2 = stego_array[row + 1, col].copy()
                            used_positions.add(position_key)
                        else:
                            continue
                    else:
                        continue

                    new_pixel1 = pixel1.copy()
                    new_pixel2 = pixel2.copy()

                    if debug and display_count < 20:
                        total_capacity = 0
                        for ch in range(3):
                            diff = abs(int(pixel1[ch]) - int(pixel2[ch]))
                            range_info = find_range(diff, range_table)
                            total_capacity += range_info[2]
                        print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {total_capacity}")
                        display_count += 1

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    lower, upper, bit_depth = range_info

                    remaining_bits = total_bits - bit_position
                    bits_to_embed = binary_message[bit_position:bit_position + min(bit_depth, remaining_bits)]
                    if len(bits_to_embed) < bit_depth:
                        bits_to_embed = bits_to_embed.ljust(bit_depth, '0')

                    new_diff = compute_new_difference(diff, bits_to_embed, range_info)
                    new_p1, new_p2 = adjust_pixel_values(pixel1[channel_idx], pixel2[channel_idx], new_diff)

                    new_pixel1[channel_idx] = new_p1
                    new_pixel2[channel_idx] = new_p2

                    if direction == 'horizontal':
                        stego_array[row, col] = new_pixel1
                        stego_array[row, col + 1] = new_pixel2
                    elif direction == 'vertical':
                        stego_array[row, col] = new_pixel1
                        stego_array[row + 1, col] = new_pixel2

                    channel_key = channel_name.lower()
                    channel_usage[channel_key] += 1
                    bits_per_channel[channel_key] += len(bits_to_embed)
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits_to_embed)
                    ])

                    bit_position += bit_depth
                    embedded_locations.append((row, col))
            else:
                # Handle diagonal direction with two passes
                # Pass 1: row % 2 == 0, col % 2 == 0
                for row, col in poi_positions:
                    if bit_position >= total_bits:
                        break

                    if not (row % 2 == 0 and col % 2 == 0):
                        continue

                    if row + 1 >= height or col + 1 >= width:
                        continue

                    position_key = (row, col, direction)
                    if position_key in used_positions:
                        continue

                    pixel1 = stego_array[row, col].copy()
                    pixel2 = stego_array[row + 1, col + 1].copy()
                    used_positions.add(position_key)

                    new_pixel1 = pixel1.copy()
                    new_pixel2 = pixel2.copy()

                    if debug and display_count < 20:
                        total_capacity = 0
                        for ch in range(3):
                            diff = abs(int(pixel1[ch]) - int(pixel2[ch]))
                            range_info = find_range(diff, range_table)
                            total_capacity += range_info[2]
                        print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {total_capacity}")
                        display_count += 1

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    lower, upper, bit_depth = range_info

                    remaining_bits = total_bits - bit_position
                    bits_to_embed = binary_message[bit_position:bit_position + min(bit_depth, remaining_bits)]
                    if len(bits_to_embed) < bit_depth:
                        bits_to_embed = bits_to_embed.ljust(bit_depth, '0')

                    new_diff = compute_new_difference(diff, bits_to_embed, range_info)
                    new_p1, new_p2 = adjust_pixel_values(pixel1[channel_idx], pixel2[channel_idx], new_diff)

                    new_pixel1[channel_idx] = new_p1
                    new_pixel2[channel_idx] = new_p2

                    stego_array[row, col] = new_pixel1
                    stego_array[row + 1, col + 1] = new_pixel2

                    channel_key = channel_name.lower()
                    channel_usage[channel_key] += 1
                    bits_per_channel[channel_key] += len(bits_to_embed)
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits_to_embed)
                    ])

                    bit_position += bit_depth
                    embedded_locations.append((row, col))

                # Pass 2: row % 2 == 1, col % 2 == 1
                for row, col in poi_positions:
                    if bit_position >= total_bits:
                        break

                    if not (row % 2 == 1 and col % 2 == 1):
                        continue

                    if row + 1 >= height or col + 1 >= width:
                        continue

                    position_key = (row, col, direction)
                    if position_key in used_positions:
                        continue

                    pixel1 = stego_array[row, col].copy()
                    pixel2 = stego_array[row + 1, col + 1].copy()
                    used_positions.add(position_key)

                    new_pixel1 = pixel1.copy()
                    new_pixel2 = pixel2.copy()

                    if debug and display_count < 20:
                        total_capacity = 0
                        for ch in range(3):
                            diff = abs(int(pixel1[ch]) - int(pixel2[ch]))
                            range_info = find_range(diff, range_table)
                            total_capacity += range_info[2]
                        print(f"Encoding at ({row},{col}): {pixel1}, {pixel2} | Capacity: {total_capacity}")
                        display_count += 1

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    lower, upper, bit_depth = range_info

                    remaining_bits = total_bits - bit_position
                    bits_to_embed = binary_message[bit_position:bit_position + min(bit_depth, remaining_bits)]
                    if len(bits_to_embed) < bit_depth:
                        bits_to_embed = bits_to_embed.ljust(bit_depth, '0')

                    new_diff = compute_new_difference(diff, bits_to_embed, range_info)
                    new_p1, new_p2 = adjust_pixel_values(pixel1[channel_idx], pixel2[channel_idx], new_diff)

                    new_pixel1[channel_idx] = new_p1
                    new_pixel2[channel_idx] = new_p2

                    stego_array[row, col] = new_pixel1
                    stego_array[row + 1, col + 1] = new_pixel2

                    channel_key = channel_name.lower()
                    channel_usage[channel_key] += 1
                    bits_per_channel[channel_key] += len(bits_to_embed)
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits_to_embed)
                    ])

                    bit_position += bit_depth
                    embedded_locations.append((row, col))

        if bit_position < total_bits:
            raise ValueError(f"Image capacity insufficient. Embedded {bit_position}/{total_bits} bits")

        # Display encoding details
        if debug:
            headers = ["Channel", "Pixel Position", "Pixel Values (P1,P2)", "Difference", "Bits Embedded"]
            print("\nEncoding Details Table:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))

            print("\nChannel Usage Statistics:")
            print(f"Red Channel: {channel_usage['red']} pixels used, {bits_per_channel['red']} bits embedded")
            print(f"Green Channel: {channel_usage['green']} pixels used, {bits_per_channel['green']} bits embedded")
            print(f"Blue Channel: {channel_usage['blue']} pixels used, {bits_per_channel['blue']} bits embedded")

        print(f"Number of pixel pairs used: {len(embedded_locations)}")
        return Image.fromarray(stego_array.astype('uint8'))

    except Exception as e:
        raise Exception(f"ADPVD encoding failed: {str(e)}")

# The decode_image_adpvd function remains unchanged for this task
def decode_image_adpvd(stego_image, debug=False):
    try:
        stego_array = np.array(stego_image.convert("RGB"))
        height, width, _ = stego_array.shape

        # Calculate capacities to determine directions
        range_table = get_range_table()
        directions = ['horizontal', 'vertical', 'diagonal']
        channels = ['Red', 'Green', 'Blue']
        capacities = {channel: {direction: 0 for direction in directions} for channel in channels}

        for channel_idx, channel_name in enumerate(channels):
            for direction in directions:
                capacity = calculate_direction_capacity(stego_array, direction, channel_idx, height, width, range_table)
                capacities[channel_name][direction] = capacity

        # Determine the best direction for each channel
        best_directions = {}
        for channel in channels:
            best_direction = max(capacities[channel], key=capacities[channel].get)
            best_directions[channel] = best_direction
        print("\nBest Directions for Each Channel (Decoding):")
        for channel, direction in best_directions.items():
            print(f"{channel}: {direction.capitalize()}")

        # POI identification
        hog_features = compute_HOG(stego_array, debug)
        threshold = compute_threshold(hog_features, debug)
        poi_indices = identify_POI(hog_features, threshold, debug)

        if len(poi_indices) == 0:
            raise ValueError("No POI indices found. Decoding failed.")

        print(f"Decoding POI indices: {poi_indices[:10]}")

        extracted_bits = ""
        terminator_found = False
        extracted_locations = []
        used_positions = set()
        table_data = []

        # Convert POI indices to (row, col) pairs
        poi_positions = [(divmod(poi, width)) for poi in poi_indices]

        for channel_idx, channel_name in enumerate(channels):
            direction = best_directions[channel_name]
            if direction != 'diagonal':
                # Handle horizontal and vertical as before
                for row, col in poi_positions:
                    pixel1, pixel2 = None, None
                    position_key = None

                    if direction == 'horizontal' and col + 1 < width:
                        if col % 2 == 0:
                            position_key = (row, col, direction)
                            if position_key in used_positions:
                                continue
                            pixel1 = stego_array[row, col]
                            pixel2 = stego_array[row, col + 1]
                            used_positions.add(position_key)
                        else:
                            continue
                    elif direction == 'vertical' and row + 1 < height:
                        if row % 2 == 0:
                            position_key = (row, col, direction)
                            if position_key in used_positions:
                                continue
                            pixel1 = stego_array[row, col]
                            pixel2 = stego_array[row + 1, col]
                            used_positions.add(position_key)
                        else:
                            continue
                    else:
                        continue

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    bits = extract_from_difference(diff, range_info)
                    extracted_bits += bits

                    channel_name = ['Red', 'Green', 'Blue'][channel_idx]
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits)
                    ])

                    if debug:
                        print(f"Channel {channel_idx}: diff={diff}, range={range_info}, extracted: {bits}")

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
            else:
                # Handle diagonal direction with two passes
                # Pass 1: row % 2 == 0, col % 2 == 0
                for row, col in poi_positions:
                    if terminator_found:
                        break

                    if not (row % 2 == 0 and col % 2 == 0):
                        continue

                    if row + 1 >= height or col + 1 >= width:
                        continue

                    position_key = (row, col, direction)
                    if position_key in used_positions:
                        continue

                    pixel1 = stego_array[row, col]
                    pixel2 = stego_array[row + 1, col + 1]
                    used_positions.add(position_key)

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    bits = extract_from_difference(diff, range_info)
                    extracted_bits += bits

                    channel_name = ['Red', 'Green', 'Blue'][channel_idx]
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits)
                    ])

                    if debug:
                        print(f"Channel {channel_idx}: diff={diff}, range={range_info}, extracted: {bits}")

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

                # Pass 2: row % 2 == 1, col % 2 == 1
                for row, col in poi_positions:
                    if terminator_found:
                        break

                    if not (row % 2 == 1 and col % 2 == 1):
                        continue

                    if row + 1 >= height or col + 1 >= width:
                        continue

                    position_key = (row, col, direction)
                    if position_key in used_positions:
                        continue

                    pixel1 = stego_array[row, col]
                    pixel2 = stego_array[row + 1, col + 1]
                    used_positions.add(position_key)

                    diff = abs(int(pixel1[channel_idx]) - int(pixel2[channel_idx]))
                    range_info = find_range(diff, range_table)
                    bits = extract_from_difference(diff, range_info)
                    extracted_bits += bits

                    channel_name = ['Red', 'Green', 'Blue'][channel_idx]
                    table_data.append([
                        channel_name,
                        f"({row},{col})",
                        f"{pixel1[channel_idx]},{pixel2[channel_idx]}",
                        diff,
                        len(bits)
                    ])

                    if debug:
                        print(f"Channel {channel_idx}: diff={diff}, range={range_info}, extracted: {bits}")

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
                    if 32 <= char_val <= 126:
                        message += chr(char_val)
                    elif debug:
                        print(f"Skipping non-printable character: {char_val} from bits {char_bits}")
                except ValueError as e:
                    if debug:
                        print(f"Error converting bits to character: {e}")
                    continue

        return message

    except Exception as e:
        raise Exception(f"ADPVD decoding failed: {str(e)}")
