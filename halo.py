#import rawpy
import numpy as np
import cv2

def circular_kernel(diameter):
    radius = diameter // 2
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return kernel

def compute_kernel_diameter(area):
    min_diameter = 10
    max_diameter = 100
    scale_factor = 0.8  # Adjust this factor to scale the kernel size 
    diameter = int(np.clip(area * scale_factor, min_diameter, max_diameter))
    return diameter

def get_dominant_color(img, mask):
    mask = (mask * 255).astype(np.uint8)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    mean_color = cv2.mean(masked_img, mask=mask)[:3]
    return np.array(mean_color)

def add_halation(input):
    # Load the .RW2 image
    #raw = rawpy.imread(input)
    #custom_wb = raw.camera_whitebalance

    # Post-process the RAW image with custom white balance
    #rgb = raw.postprocess(output_bps=16, user_wb=custom_wb)

    # Convert to float32 numpy array for better precision during processing
    #img = rgb.astype(np.float32) / 65535.0

    # Convert to grayscale
    gray = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)

    # Apply a threshold to get bright areas
    _, bright_mask = cv2.threshold(gray, 0.9999, 1.0, cv2.THRESH_BINARY)

    # Find contours of the bright areas
    contours, _ = cv2.findContours(bright_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask of the bright areas, ignoring small areas
    mask = np.zeros_like(gray)
    min_contour_area = 500  # Minimum area threshold to consider a contour for halation

    mask_dilated_combined = np.zeros_like(gray)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_contour_area:

            individual_mask = np.zeros_like(gray)
            cv2.drawContours(individual_mask, [contour], -1, (1), thickness=cv2.FILLED)

            # Compute kernel diameter based on the area of the contour
            kernel_diameter = compute_kernel_diameter(area)

            # Dilate the individual mask
            kernel = circular_kernel(kernel_diameter)
            individual_mask_dilated = cv2.dilate(individual_mask, kernel, iterations=1) 

            # Combine the dilated mask with the accumulated dilated masks
            mask_dilated_combined = np.maximum(mask_dilated_combined, individual_mask_dilated)

    # Apply Gaussian Blur
    glow = cv2.GaussianBlur(mask_dilated_combined, (0, 0), sigmaX=10, sigmaY=10)


    dominant_color = get_dominant_color(input, mask_dilated_combined)

    # Adjust the tint color based on the dominant color
    red_tint = np.array([0.6, 0.15, 0])  # Base red-orange color for halation
    adjusted_tint = 0.5 * red_tint + 0.5 * (dominant_color / 255.0)  # Blend with the dominant color

    # Create the glow effect with the adjusted tint color
    glow_rgb = cv2.merge([glow * adjusted_tint[0], glow * adjusted_tint[1], glow * adjusted_tint[2]])

    # Amplify the brightness of the bright areas
    bright_areas = input * mask[..., np.newaxis]
    bright_areas = np.clip(bright_areas * 2.0, 0, 1)  # Increase brightness

    # Combine the amplified bright areas with the glow effect
    enhanced_glow_rgb = np.clip(bright_areas + glow_rgb, 0, 1)

    # Combine the original image with the enhanced bright areas and glow effect
    halation = np.clip(input + enhanced_glow_rgb, 0, 1)

    # Save the output image
    return halation

# Example usage
#add_halation('raw.RW2', 'output.jpg')
