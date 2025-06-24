import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import glob

def detect_and_compute(image):
    """Detect keypoints and descriptors using SIFT"""
    if image is None:
        raise ValueError("Image is None in detect_and_compute")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        print("Warning: No descriptors found in image")
        return [], None
    return kp, des

def match_features_hungarian(des1, des2, max_distance=0.10):
    """Match descriptors using Hungarian Algorithm"""
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Warning: Empty descriptors in match_features_hungarian")
        return []

    # Compute cost matrix (cosine or euclidean)
    cost_matrix = cdist(des1, des2, metric='cosine')  # shape (N1, N2)

    # Solve assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Filter matches by a distance threshold (optional)
    matches = []
    for i, j in zip(row_ind, col_ind):
        dist = cost_matrix[i, j]
        if dist <= max_distance:  # reject bad matches
            match = cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=dist)
            matches.append(match)

    return matches


def match_features_greedy(des1, des2):
    """Match features using a greedy approach with cosine distance"""
    distances = cdist(des1, des2, metric='cosine')
    matches = []
    used_des2 = set()

    for i in range(len(des1)):
        j = np.argmin(distances[i])
        if j not in used_des2:
            matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=distances[i][j]))
            used_des2.add(j)
    return matches

def match_features(des1, des2, ratio=0.75):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        print("Warning: Empty descriptors in match_features")
        return []
    
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    matches = flann.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches

def compute_homography(kp1, kp2, matches, reproj_thresh=5.0):
    if len(matches) < 4:
        print(f"Warning: Only {len(matches)} matches - need at least 4 for homography")
        raise ValueError("Sybau")

    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
    return H, mask
        

def stitch_images(images):
    if len(images) < 2:
        raise ValueError("Need at least 2 images to stitch")
    
    # Initialize
    panorama = images[0].copy()
    
    kp1, des1 = detect_and_compute(panorama)
    
    for i in range(1, len(images)):
        print(f"\nStitching image {i+1}/{len(images)}")
        image2 = images[i]
        print(f"Image size: {image2.shape[1]}x{image2.shape[0]}")
        
        # Detect and match features
        kp2, des2 = detect_and_compute(image2)
        if not kp2 or des2 is None:
            print("Skipping image - no features detected")
            continue
            
        print(f"Features found: {len(kp2)}")
        matches = match_features(des1, des2)
        print(f"Good matches: {len(matches)}")
        
        if len(matches) < 10:
            print(f"Warning: Only {len(matches)} matches found - skipping image")
            continue
        
        # merging - homography
        H, mask = compute_homography(kp1, kp2, matches)
        
        if H is None:
            print("Homography computation failed - skipping image")
            continue
            
        print(f"Homography matrix:\n{H}")
        
        # Calculate output dimensions
        h1, w1 = panorama.shape[:2]
        h2, w2 = image2.shape[:2]
        
        # Find corners
        corners = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        
        # Combine 
        all_corners = np.concatenate((
            np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2),
            warped_corners
        ))
        
        # Find dimensions
        min_vals = all_corners.min(axis=0).ravel()
        max_vals = all_corners.max(axis=0).ravel()
        
        x_min, y_min = min_vals
        x_max, y_max = max_vals
        
        # Tune dimenstions
        x_min, y_min = int(np.floor(x_min)) - 1, int(np.floor(y_min)) - 1
        x_max, y_max = int(np.ceil(x_max)) + 1, int(np.ceil(y_max)) + 1
        
        # Matrix opearion
        tx, ty = -x_min, -y_min
        translation = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)
        
        # Warp the new image
        output_shape = (x_max - x_min, y_max - y_min)
        print(f"Output size: {output_shape[0]}x{output_shape[1]}")
        
        if output_shape[0] <= 0 or output_shape[1] <= 0:
            print(f"Error: Invalid output shape {output_shape}")
            continue
            
        # Warp new image with combined transformation
        warped_img2 = cv2.warpPerspective(image2, translation @ H, output_shape)
        
        # Warp the existing panorama
        warped_pano = cv2.warpPerspective(panorama, translation, output_shape)
        
        # Create result canvas
        result = np.zeros((output_shape[1], output_shape[0], 3), dtype=np.uint8)
        
        # Create masks for blending
        mask_pano = np.sum(warped_pano, axis=2) > 0
        mask_img2 = np.sum(warped_img2, axis=2) > 0
        overlap = mask_pano & mask_img2
        
        # Place non-overlapping regions
        result[mask_pano] = warped_pano[mask_pano]
        result[mask_img2 & ~overlap] = warped_img2[mask_img2 & ~overlap]
        
        # Blend overlapping regions
        if overlap.any():
            # Simple average blending
            result[overlap] = (warped_pano[overlap] * 0.5 + warped_img2[overlap] * 0.5).astype(np.uint8)
        else:
            print("Warning: No overlap detected - using panorama")
            result = warped_pano
        
        panorama = result
        print(f"New panorama size: {panorama.shape[1]}x{panorama.shape[0]}")
        
        # Update features for next iteration (from the new panorama)
        kp1, des1 = detect_and_compute(panorama)
    
    return panorama

# Load images
image_paths = sorted(glob.glob("images/*.jpg"))
print(f"Found {len(image_paths)} images")

if not image_paths:
    print("No images found in images/ folder")
    exit()

images = []
for i, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is not None:
        print(f"Loaded image {i+1}: {path} ({img.shape[1]}x{img.shape[0]})")

        # Resize
        max_dim = max(img.shape[1], img.shape[0])
        if max_dim > 1000:
            scale = 1000 / max_dim
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            print(f"  Resized to: {img.shape[1]}x{img.shape[0]}")
        images.append(img)
    else:
        print(f"Error loading image: {path}")

if len(images) < 2:
    print("Need at least 2 valid images")
    exit()

# Stitch images
print(f"\nStarting stitching of {len(images)} images...")
try:
    result = stitch_images(images)
    if result is not None:
        cv2.imwrite("result.jpg", result)
        print("Stitching completed successfully, result saved as result.jpg")

        # Show result
        cv2.imshow("result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed - no result")
except Exception as e:
    print(f"Error during stitching: {e}")
    import traceback
    traceback.print_exc()