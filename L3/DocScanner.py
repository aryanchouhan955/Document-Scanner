import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def order_points(pts):
    """Order 4 points: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    """Apply perspective transform to obtain a top-down view using 4 corner pts."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.hypot(br[0] - bl[0], br[1] - bl[1])
    widthB = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.hypot(tr[0] - br[0], tr[1] - br[1])
    heightB = np.hypot(tl[0] - bl[0], tl[1] - bl[1])
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_canny(gray, sigma=0.33):
    """Automatic Canny thresholds using median of the grayscale image."""
    med = np.median(gray)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    return cv.Canny(gray, lower, upper)

def four_corners_document(img: np.ndarray, procesing_height_in_px: int = 800):

    """
    Detect four document corners and return them as float32 points in original image coords:
        array([[x0, y0],
               [x1, y1],
               [x2, y2],
               [x3, y3]], dtype=float32)
    Returns None if no contours found.
    """
    # ---------- Input checks ----------
    if img is None:
        raise ValueError("Input image is None")

    # ---------- 1) Resize for speed ----------
    h, w = img.shape[:2]
    ratio = min(1, procesing_height_in_px/h)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, procesing_height_in_px)
    resized_img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)

    # ---------- 2) Convert to gray ----------
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)

     # ---------- 3) Gaussian blur to reduce noise ----------
    blur = cv.GaussianBlur(gray, (5, 5), 0)

    # ---------- 4) Edge detection ----------
    edges = auto_canny(blur)

    # ---------- 5) Morphological closing to bridge gaps ----------
    # Dilation: Expands white (foreground) regions in the image.
    # → Makes objects thicker, fills small black holes/gaps.
    
    # Erosion: Shrinks white regions.
    # → Removes noise and extra pixels added by dilation.
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

    
    
    
    
    # ---------- 7) Find contours (compatibility across OpenCV versions) ----------
    contours, _ = cv.findContours(closed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # If no contours found, nothing to do
    if not contours:
        return None

    # ---------- 8) Sort contours by area and keep top candidates ----------
    contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

    # ---------- 9) Find contour that approximates to 4 points ----------
    doc_cnt = None
    for c in contours:
        peri = cv.arcLength(c, True)                      # contour perimeter
        approx = cv.approxPolyDP(c, 0.02 * peri, True)    # approx polygon
        if len(approx) == 4:                              # found 4-corner contour
            doc_cnt = approx
            break

    # ---------- 10) Fallback: minimum-area rectangle of the largest contour ----------
    if doc_cnt is None:
        c = contours[0]  # we know contours is non-empty
        rect = cv.minAreaRect(c)           # (center (x,y), (w,h), angle)
        box = cv.boxPoints(rect)           # 4 corners float32
        doc_cnt = np.array(box, dtype=np.float32).reshape((4, 1, 2))

    # ---------- 11) Convert to (4,2) float32 and scale back to original image size ----------
    pts = doc_cnt.reshape(4, 2).astype("float32")

    pts /= float(ratio)

    return pts

def crop_doc(img: np.ndarray):
    """Detect and crop document. """
    corners = four_corners_document(img)
    if corners is None:
        raise RuntimeError("No document detected.")
    return four_point_transform(img, corners)
    

if __name__ == '__main__':
# --- Read input image ---
    path_image = r"L3\input2.jpg"
    img = cv.imread(path_image)

    if img is None:
        raise FileNotFoundError("Image not found at specified path")

    # --- Crop document using your function ---
    doc = crop_doc(img)


    # --- Convert BGR to RGB for correct color display in matplotlib ---
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    doc_rgb = cv.cvtColor(doc, cv.COLOR_BGR2RGB)

    # --- Display both images side by side ---
    plt.figure(figsize=(12, 6))  # width, height in inches

    # Left: Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")

    # Right: Cropped Document
    plt.subplot(1, 2, 2)
    plt.imshow(doc_rgb)
    plt.title("Cropped Document")
    plt.axis("off")

    # --- Show the figure ---
    plt.tight_layout()
    plt.show()