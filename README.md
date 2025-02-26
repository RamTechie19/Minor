# Steganography using P-ADPVD Technique

This project implements a steganography technique based on **P-ADPVD** (Pixel Adaptive Data Processing for Visual Data Hiding). The method embeds and extracts hidden messages within images using **HOG (Histogram of Oriented Gradients)** to determine areas for data encoding.

## How It Works

### 1. Encoding (Hiding a message)
- Converts the input message into **binary format**.
- Identifies **Points of Interest (POI)** in the image using **HOG feature extraction** and **Threshold Computation** .
- Embeds the message into **specific pixel pairs** 
- Saves the **stego image** containing the hidden message.

### 2. Decoding (Extracting the message)
- Loads the **stego image**.
- Identifies the **POI locations** used during encoding.
- Extracts the **hidden binary data** from the pixels.
- Converts the binary back into **readable text**.

## Features
  
✅ **Histogram of Oriented Gradients (HOG) for POI detection**  
✅ **Adaptive thresholding for optimal embedding**  
✅ **Binary message embedding and extraction**  
✅ **Debug mode for in-depth analysis**  

## Technologies Used

- **Programming Language:** Python  
- **Libraries:** NumPy, OpenCV, PIL  

## File Descriptions

- **`main.py`** - Command-line interface to encode or decode messages in images.
- **`encoder.py`** - Implements encoding using POI-based LSB embedding.
- **`decoder.py`** - Extracts hidden messages from stego images.
- **`utils.py`** - Utility functions for HOG computation, POI identification, and bit embedding/extraction.

## Installation & Usage

1. **Clone the repository**  
   ```bash
   git clone https://github.com/RamTechie19/Minor.git
   cd Minor
   ```

2. **Install dependencies**  
   ```bash
   pip install numpy opencv-python pillow
   ```

3. **Encoding a message into an image**  
   ```bash
   python main.py encode input_image.png --message "Hidden text" --output stego_image.png
   ```

4. **Decoding a message from an image**  
   ```bash
   python main.py decode stego_image.png
   ```

## Future Improvements

- Implement **AES encryption** for added security   
- Optimize **message capacity and robustness**  
- Create a **GUI for user-friendly interaction**  

## Contributing

Contributions are welcome! Feel free to fork the repository, submit pull requests, or report issues.

## Contact

For queries or collaboration, reach out to me at [your email] or visit my GitHub profile [@RamTechie19](https://github.com/RamTechie19).

---

🚀 **Enhance security through steganography!**
