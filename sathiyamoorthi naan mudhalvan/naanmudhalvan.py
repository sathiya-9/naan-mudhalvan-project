import cv2
import numpy as np

def detect_defects(image_path, use_canny=True, use_equalization=False, window_title="Defect Detection"):
    # Load the input image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image '{image_path}' not found.")
        return

    # Resize image for consistency
    image = cv2.resize(image, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization if needed (useful for metal surfaces)
    if use_equalization:
        gray = cv2.equalizeHist(gray)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Choose processing technique
    if use_canny:
        # Canny edge detection (better for metal cracks)
        edges = cv2.Canny(blurred, 100, 200)

        # Morphological dilation to thicken edges
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.dilate(edges, kernel, iterations=1)
    else:
        # Simple binary thresholding (better for concrete cracks)
        _, processed = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around defects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Ignore small areas
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Defect", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show result
    cv2.imshow(window_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==== Run Detection for Both Types ====

# 1. Concrete Crack Detection (e.g., 'concrete.jpg')
detect_defects("input.jpg", use_canny=False, use_equalization=False, window_title="Concrete Defect Detection")

# 2. Metal Crack Detection (e.g., 'metal.jpg')
detect_defects("input 1.jpg", use_canny=True, use_equalization=True, window_title="Metal Crack Detection")
