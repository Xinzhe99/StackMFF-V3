# -*- coding: utf-8 -*-
# @Author  : XinZhe Xie
# @University  : ZheJiang University
# @Description : This script aligns and crops a stack of images using SIFT feature detection and homography transformation.
# It takes a folder of input images, aligns them to a reference image using SIFT keypoints and FLANN matching,
# computes homography matrices for alignment, determines a common cropping region, and saves the aligned and cropped images.

import cv2
import numpy as np
import os
import argparse

def align_and_crop_image_stack(input_folder, output_folder):
    """
    Align and crop a stack of images using SIFT feature detection and homography transformation.
    
    Args:
        input_folder (str): Path to the folder containing input images
        output_folder (str): Path to the folder where aligned images will be saved
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

    if not image_files:
        print(f"No image files found in '{input_folder}' folder.")
        return

    # Read all images
    images = []
    for f in image_files:
        img_path = os.path.join(input_folder, f)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read image: {img_path}")
            continue
        images.append(img)

    if not images:
        print("Failed to load any images, script terminated.")
        return

    print(f"Loaded {len(images)} images.")

    # Use the first image as reference image
    reference_image = images[0]
    aligned_images = [reference_image]

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find SIFT keypoints and descriptors for reference image
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY), None)

    # Store transformation matrices for all images
    M_matrices = [np.eye(3, 3)] # Transformation matrix for reference image is identity matrix

    print("Starting image alignment...")

    for i in range(1, len(images)):
        print(f"Aligning image {i+1}/{len(images)}...")
        current_image = images[i]
        
        # Find SIFT keypoints and descriptors for current image
        kp2, des2 = sift.detectAndCompute(cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY), None)

        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print(f"Warning: Insufficient keypoints in image {i+1}, skipping alignment.")
            aligned_images.append(current_image)
            M_matrices.append(np.eye(3, 3))
            continue

        matches = flann.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) > 10:  # Need at least 10 good matches to compute homography matrix
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0) # Note: here it's dst_pts to src_pts
            if M is None:
                print(f"Warning: Could not compute homography matrix for image {i+1}, skipping alignment.")
                aligned_images.append(current_image)
                M_matrices.append(np.eye(3, 3))
                continue
            
            h, w, _ = reference_image.shape
            aligned_image = cv2.warpPerspective(current_image, M, (w, h))
            aligned_images.append(aligned_image)
            M_matrices.append(M)
        else:
            print(f"Warning: Insufficient matching points for image {i+1}, skipping alignment.")
            aligned_images.append(current_image)
            M_matrices.append(np.eye(3, 3)) # If not aligned, use identity matrix


    # Calculate cropping region for all images
    min_x, min_y = 0, 0
    max_x, max_y = reference_image.shape[1], reference_image.shape[0]

    print("Calculating cropping region...")

    for i, M in enumerate(M_matrices):
        if i == 0: continue # Skip reference image as it has no transformation

        h, w, _ = images[i].shape
        
        # Transform four corners to find image position in reference coordinate system
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, M)
        
        # Find boundaries of all images in reference coordinate system
        min_x = max(min_x, int(np.min(transformed_corners[:, 0, 0])))
        min_y = max(min_y, int(np.min(transformed_corners[:, 0, 1])))
        max_x = min(max_x, int(np.max(transformed_corners[:, 0, 0])))
        max_y = min(max_y, int(np.max(transformed_corners[:, 0, 1])))
    
    # Ensure cropping region is valid
    crop_x = max(0, min_x)
    crop_y = max(0, min_y)
    crop_width = min(reference_image.shape[1] - crop_x, max_x - min_x)
    crop_height = min(reference_image.shape[0] - crop_y, max_y - min_y)

    if crop_width <= 0 or crop_height <= 0:
        print("Warning: Could not calculate valid cropping region. All images may lack overlap or alignment failed. Saving uncropped images.")
        crop_x, crop_y = 0, 0
        crop_width, crop_height = reference_image.shape[1], reference_image.shape[0]
    else:
        print(f"Cropping region: x={crop_x}, y={crop_y}, width={crop_width}, height={crop_height}")


    print("Saving aligned and cropped images...")
    # Save aligned and cropped images
    for i, img in enumerate(aligned_images):
        cropped_img = img[crop_y : crop_y + crop_height, crop_x : crop_x + crop_width]
        output_path = os.path.join(output_folder, f"aligned_{image_files[i]}")
        cv2.imwrite(output_path, cropped_img)
        print(f"Saved: {output_path}")

    print("All images processed.")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Align and crop a stack of images using SIFT feature detection and homography transformation.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input folder containing images to align")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder where aligned images will be saved")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the alignment and cropping function
    align_and_crop_image_stack(args.input, args.output)