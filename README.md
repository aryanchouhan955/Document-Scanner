# DocScanner || Digital Image Processing || IIIT Bhopal

DocScanner is a computer vision project built with Python and OpenCV that demonstrates fundamental image processing techniques, culminating in a functional document scanning pipeline. The project is structured progressively across three levels, moving from low-level custom implementations of filters to a higher-level application using optimized OpenCV functions.

## What is this project?

This project serves as both an educational journey into computer vision and a practical tool. It is divided into three main components:

- **Level 1 (L1) - Image Blurring**: Explores spatial filtering by implementing Averaging (Mean) and Gaussian blur algorithms from scratch using manual convolution loops.
- **Level 2 (L2) - Edge Detection**: Focuses on finding edges using First-Order (Roberts, Prewitt, Sobel) and Second-Order (Laplacian) gradients, again utilizing manual convolution to deeply understand the math behind edge detection.
- **Level 3 (L3) - Document Scanner**: The core application. It uses a robust pipeline combining OpenCV's optimized functions to detect a document in an image, extract its corners, and warp it into a flat, top-down view.

Jupyter Notebook equivalents of these scripts are also available in the `JupyterFile` directory for interactive exploration.

## Why does it exist?

DocScanner was created to bridge the gap between theoretical image processing concepts and practical applications. While it's easy to call `cv2.GaussianBlur` or `cv2.Canny`, implementing these algorithms from scratch (as seen in L1 and L2) builds a stronger foundational understanding. The project then takes these concepts and applies them to a real-world problem: extracting a document from a cluttered background (L3). 

## How it works (The Scanning Pipeline)

The final document scanner (`main/L3/DocScanner.py`) operates through the following pipeline:

1.  **Preprocessing**: The input image is resized to speed up processing. It is then converted to grayscale and smoothed with a Gaussian blur to reduce noise.
2.  **Edge Detection**: An automatic Canny edge detector is applied. The thresholds are dynamically calculated based on the median pixel intensity of the image.
3.  **Morphological Operations**: Morphological closing (dilation followed by erosion) is used to close small gaps in the detected edges, ensuring continuous boundaries for the document.
4.  **Contour Detection**: The script finds contours in the edge map and sorts them by area, keeping only the largest ones.
5.  **Corner Approximation**: It approximates the largest contours to polygons. It looks for a shape with exactly 4 points (a quadrilateral). If found, these are assumed to be the document corners. As a fallback, it uses a minimum-area bounding rectangle.
6.  **Perspective Transform**: The 4 detected corners are ordered (top-left, top-right, bottom-right, bottom-left). A perspective transform is then calculated and applied to warp the angled document into a perfect top-down view.

## Setup and Usage

### Prerequisites
- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)

### Running the Code
Navigate to the specific level directory and run the python script. Ensure the input image paths within the scripts are correctly pointing to your test images.

```bash
# Example: Running the document scanner
python main/L3/DocScanner.py
```

## Future Improvements

- **Optical Character Recognition (OCR)**: Integrate Tesseract (or similar tools) to extract text directly from the scanned documents.
- **Image Enhancement**: Add post-processing steps such as adaptive thresholding (binarization), contrast enhancement, and shadow removal to produce "scan-like" black and white documents.
- **User Interface (GUI)**: Develop a simple graphical interface using Tkinter, PyQt, or a web framework (like Gradio/Streamlit) to make the tool more accessible without modifying code.
- **Batch Processing**: Add support for processing entire directories of images simultaneously.
- **Performance Optimization**: The manual convolutions in L1 and L2 are educational but slow. They could be optimized using NumPy's vectorized operations (`scipy.signal.convolve2d`) for faster execution.
- **Robustness under varying lighting**: Improve the contour detection pipeline to be more resilient against harsh shadows or very low contrast backgrounds.
