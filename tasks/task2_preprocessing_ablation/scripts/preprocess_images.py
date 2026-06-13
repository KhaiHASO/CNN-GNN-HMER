import os
import shutil
import cv2
import numpy as np

def crop_formula_region(img_gray, padding=10):
    """Crop the formula region from the image by removing background margins."""
    # Determine if background is light or dark
    # We check the average of the edges
    h, w = img_gray.shape
    edge_pixels = np.concatenate([
        img_gray[0, :], img_gray[-1, :],
        img_gray[:, 0], img_gray[:, -1]
    ])
    bg_color = np.median(edge_pixels)
    
    # Binarize to find foreground
    if bg_color > 127:
        # Light background, dark text
        _, thresh = cv2.threshold(img_gray, bg_color - 30, 255, cv2.THRESH_BINARY_INV)
    else:
        # Dark background, light text
        _, thresh = cv2.threshold(img_gray, bg_color + 30, 255, cv2.THRESH_BINARY)
        
    # Find coordinates of all non-zero pixels
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w_box, h_box = cv2.boundingRect(coords)
        # Add padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + w_box + padding)
        y2 = min(h, y + h_box + padding)
        return img_gray[y1:y2, x1:x2]
    return img_gray

def resize_keep_aspect(img, target_h=192, max_w=672):
    """Resize image to target height while keeping aspect ratio, up to max width."""
    h, w = img.shape[:2]
    scale = target_h / h
    new_w = int(w * scale)
    if new_w > max_w:
        new_w = max_w
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

def preprocess_p1(img_path):
    """P1: Gray -> Crop -> Resize keeping aspect ratio."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = crop_formula_region(gray, padding=8)
    resized = resize_keep_aspect(cropped, target_h=192, max_w=672)
    # Convert back to BGR for consistent channel count
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

def preprocess_p2(img_path):
    """P2: P1 + Threshold & Denoise."""
    img = cv2.imread(img_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped = crop_formula_region(gray, padding=8)
    
    # Otsu thresholding for crisp text
    # Assuming light background, if dark background we invert
    h, w = cropped.shape
    edge_pixels = np.concatenate([cropped[0, :], cropped[-1, :], cropped[:, 0], cropped[:, -1]])
    edge_mean = np.mean(edge_pixels)
    
    if edge_mean > 127:
        # Light background, dark text
        _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Dark background, light text. Invert to keep black text on white background
        _, thresh = cv2.threshold(cropped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
    # Denoise using median blur to clean up stray dots
    denoised = cv2.medianBlur(thresh, 3)
    
    # Resize keeping aspect ratio
    resized = resize_keep_aspect(denoised, target_h=192, max_w=672)
    return cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

def main():
    src_dir = "tasks/task1_unimernet_baseline/data/quick_test_50/images"
    p0_dir = "tasks/task2_preprocessing_ablation/data/p0_original"
    p1_dir = "tasks/task2_preprocessing_ablation/data/p1_crop_gray_resize"
    p2_dir = "tasks/task2_preprocessing_ablation/data/p2_threshold_denoise"
    
    for d in [p0_dir, p1_dir, p2_dir]:
        os.makedirs(d, exist_ok=True)
        
    print("Starting preprocessing ablation images...")
    images = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    
    for img_name in images:
        src_path = os.path.join(src_dir, img_name)
        
        # P0: Copy original image
        shutil.copy(src_path, os.path.join(p0_dir, img_name))
        
        # P1: Crop + Gray + Resize
        p1_img = preprocess_p1(src_path)
        if p1_img is not None:
            cv2.imwrite(os.path.join(p1_dir, img_name), p1_img)
            
        # P2: P1 + Threshold & Denoise
        p2_img = preprocess_p2(src_path)
        if p2_img is not None:
            cv2.imwrite(os.path.join(p2_dir, img_name), p2_img)
            
    print(f"Preprocessed {len(images)} images for configs P0, P1, and P2 successfully.")

if __name__ == "__main__":
    main()
