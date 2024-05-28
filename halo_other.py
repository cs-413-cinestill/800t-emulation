import numpy as np
import cv2
import rawpy
from numba import jit, prange

# Constants
PI = 3.14159265358979

# Placeholder for UI parameters
exposure_lost = -3.0
green_exposure_lost = -1.4
blue_exposure_lost = -1.4
blur_type = "GAUSSIAN"  # Options: "GAUSSIAN", "EXPONENTIAL"
blur_amt = 3.0
correct_red_shift = "MATRIX"  # Options: "OFF", "GAIN", "MATRIX"

def main(input_image_path, output_image_path):
    # Load image using rawpy for .RW2 files
    with rawpy.imread(input_image_path) as raw:
        rgb_image = raw.postprocess()

    # Convert the rawpy output to OpenCV format
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    height, width, _ = image.shape
    p_TexR, p_TexG, p_TexB = image[:, :, 2], image[:, :, 1], image[:, :, 0]  # OpenCV uses BGR by default

    output_image = np.zeros_like(image)
    transform(width, height, p_TexR, p_TexG, p_TexB, output_image)

    # Save the processed image
    cv2.imwrite(output_image_path, output_image)

@jit
def transform(p_Width, p_Height, p_TexR, p_TexG, p_TexB, output_image):
    exposure_lost_lin = 2 ** exposure_lost
    green_exposure_lost_lin = 2 ** green_exposure_lost
    blue_exposure_lost_lin = 2 ** blue_exposure_lost
    blur_radius = int((blur_amt / 1000.0) * p_Width)

    for y in prange(p_Height):
        for x in prange(p_Width):
            X = x / (p_Width - 1) - 0.5
            Y = y / (p_Height - 1) - 0.5

            # Clamp the indices to integer values
            x_clamped = max(0, min(int(x), p_TexR.shape[1] - 1))
            y_clamped = max(0, min(int(y), p_TexR.shape[0] - 1))

            r = p_TexR[y_clamped, x_clamped]
            g = p_TexG[y_clamped, x_clamped]
            b = p_TexB[y_clamped, x_clamped]
            curr_color = np.array([r, g, b])


            scale_color = curr_color  # Just here need the blur

            # Halation function
            halation_color = np.copy(curr_color)
            halation_color[0] += scale_color[0] * exposure_lost_lin
            halation_color[1] += scale_color[0] * exposure_lost_lin * green_exposure_lost_lin
            halation_color[2] += scale_color[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin

            if correct_red_shift == "MATRIX":
                red = np.array([1.0, 0.0, 0.0])
                red_output = np.copy(red)
                red_output[0] += red[0] * exposure_lost_lin
                red_output[1] += red[0] * exposure_lost_lin * green_exposure_lost_lin
                red_output[2] += red[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin

                green = np.array([0.0, 1.0, 0.0])
                green_output = np.copy(green)
                green_output[0] += green[0] * exposure_lost_lin
                green_output[1] += green[0] * exposure_lost_lin * green_exposure_lost_lin
                green_output[2] += green[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin

                blue = np.array([0.0, 0.0, 1.0])
                blue_output = np.copy(blue)
                blue_output[0] += blue[0] * exposure_lost_lin
                blue_output[1] += blue[0] * exposure_lost_lin * green_exposure_lost_lin
                blue_output[2] += blue[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin

                forward_matrix = np.array([
                    [red_output[0], green_output[0], blue_output[0]],
                    [red_output[1], green_output[1], blue_output[1]],
                    [red_output[2], green_output[2], blue_output[2]]
                ])
                inv_matrix = np.linalg.inv(forward_matrix)
                output_color = np.dot(inv_matrix, halation_color)
            elif correct_red_shift == "GAIN":
                white = np.array([1.0, 1.0, 1.0])
                white_output = np.copy(white)
                white_output[0] += white[0] * exposure_lost_lin
                white_output[1] += white[0] * exposure_lost_lin * green_exposure_lost_lin
                white_output[2] += white[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin
                output_color = halation_color * white / white_output

            else:
                output_color = halation_color

            output_image[y, x] = output_color.clip(0, 255)

# Example usage:
input_image_path = 'raw.RW2'
output_image_path = 'output_image_2.jpg'
main(input_image_path, output_image_path)
