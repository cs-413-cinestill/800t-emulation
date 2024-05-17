import numpy as np
import cv2
import rawpy
from math import exp, hypot, pow, floor, ceil
import time

# Constants
PI = 3.14159265358979


exposure_lost = -3.0
green_exposure_lost = -1.4
blue_exposure_lost = -1.4
blur_type = "GAUSSIAN"  # Options: "GAUSSIAN", "EXPONENTIAL"
blur_amt = 3.0
correct_red_shift = "MATRIX"  # Options: "OFF", "GAIN", "MATRIX"

# Helper functions
def powf(base, exp):
    return pow(abs(base), exp) if base >= 0 else -pow(abs(base), exp)

def mv_33_3(mat, v):
    return np.dot(mat, v)

def mat_inverse_33(m):
    return np.linalg.inv(m)

def get_coord(x, width):
    return int(round((x + 0.5) * (width + 1)))

def get_coord_float(x, width):
    return (x + 0.5) * (width + 1)

def get_color(x, y, p_TexR, p_TexG, p_TexB):
    # Clamp coordinates to image boundaries
    x = max(0, min(x, p_TexR.shape[1] - 1))
    y = max(0, min(y, p_TexR.shape[0] - 1))

    r = p_TexR[y, x]
    g = p_TexG[y, x]
    b = p_TexB[y, x]
    return np.array([r, g, b])

def sample_point_bilinear(x, y, p_Width, p_Height, p_TexR, p_TexG, p_TexB):
    f_x = get_coord_float(x, p_Width)
    f_y = get_coord_float(y, p_Height)

    x_low = int(floor(f_x))
    x_high = int(ceil(f_x))
    y_low = int(floor(f_y))
    y_high = int(ceil(f_y))

    c_ll = get_color(x_low, y_low, p_TexR, p_TexG, p_TexB)
    c_lh = get_color(x_low, y_high, p_TexR, p_TexG, p_TexB)
    c_hl = get_color(x_high, y_low, p_TexR, p_TexG, p_TexB)
    c_hh = get_color(x_high, y_high, p_TexR, p_TexG, p_TexB)

    mix_x = f_x - x_low
    mix_y = f_y - y_low

    c_l = (1 - mix_x) * c_ll + mix_x * c_hl
    c_h = (1 - mix_x) * c_lh + mix_x * c_hh
    c = (1 - mix_y) * c_l + mix_y * c_h
    return c

def convert_gamma(g):
    if g <= 0:
        return 1.0 + (-4.0 * g)
    else:
        return 1.0 / (4.0 * g + 1)

def gaussian_blur(radius, x, y, p_Width, p_Height, p_TexR, p_TexG, p_TexB):
    std = radius / 2.0
    window_size = int(ceil(2 * std * 3))

    center_x = get_coord(x, p_Width)
    center_y = get_coord(y, p_Height)

    sum = np.zeros(3)
    weight_sum = 0
    for i in range(center_x - (window_size // 2), center_x + (window_size // 2) + 1):
        for j in range(center_y - (window_size // 2), center_y + (window_size // 2) + 1):
            runner = get_color(i, j, p_TexR, p_TexG, p_TexB)
            weight = 1.0 / (2.0 * PI * std * std) * exp((pow(abs(center_x - i), 2) + pow(abs(center_y - j), 2)) / (-2.0 * std * std))
            weight_sum += weight
            sum += runner * weight
    return sum / weight_sum

def exp_k(x, y, r0):
    return exp(-hypot(x, y) / r0)

def exp_blur(radius, x, y, p_Width, p_Height, p_TexR, p_TexG, p_TexB):
    center_x = get_coord(x, p_Width)
    center_y = get_coord(y, p_Height)

    window_size = int(ceil(2 * radius * 4.5))
    sum = np.zeros(3)
    weight_sum = 0
    for i in range(center_x - (window_size // 2), center_x + (window_size // 2) + 1):
        for j in range(center_y - (window_size // 2), center_y + (window_size // 2) + 1):
            weight = exp_k(i - center_x, j - center_y, radius)
            runner = get_color(i, j, p_TexR, p_TexG, p_TexB)
            sum += weight * runner
            weight_sum += weight
    return sum / weight_sum

def halation(scene_color, diffused_color, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin):
    output = np.copy(scene_color)
    output[0] += diffused_color[0] * exposure_lost_lin
    output[1] += diffused_color[0] * exposure_lost_lin * green_exposure_lost_lin
    output[2] += diffused_color[0] * exposure_lost_lin * green_exposure_lost_lin * blue_exposure_lost_lin
    return output

def transform(p_Width, p_Height, p_X, p_Y, p_TexR, p_TexG, p_TexB):
    X = p_X / (p_Width - 1) - 0.5
    Y = p_Y / (p_Height - 1) - 0.5

    exposure_lost_lin = 2 ** exposure_lost
    green_exposure_lost_lin = 2 ** green_exposure_lost
    blue_exposure_lost_lin = 2 ** blue_exposure_lost
    blur_radius = (blur_amt / 1000.0) * p_Width

    if blur_type == "GAUSSIAN":
        scale_color = gaussian_blur(blur_radius, X, Y, p_Width, p_Height, p_TexR, p_TexG, p_TexB)
    elif blur_type == "EXPONENTIAL":
        scale_color = exp_blur(blur_radius / 3, X, Y, p_Width, p_Height, p_TexR, p_TexG, p_TexB)

    curr_color = get_color(p_X, p_Y, p_TexR, p_TexG, p_TexB)
    halation_color = halation(curr_color, scale_color, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin)

    if correct_red_shift == "MATRIX":
        red = np.array([1.0, 0.0, 0.0])
        red_output = halation(red, red, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin)
        green = np.array([0.0, 1.0, 0.0])
        green_output = halation(green, green, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin)
        blue = np.array([0.0, 0.0, 1.0])
        blue_output = halation(blue, blue, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin)

        forward_matrix = np.array([
            [red_output[0], green_output[0], blue_output[0]],
            [red_output[1], green_output[1], blue_output[1]],
            [red_output[2], green_output[2], blue_output[2]]
        ])
        inv_matrix = mat_inverse_33(forward_matrix)
        output_color = mv_33_3(inv_matrix, halation_color)
    elif correct_red_shift == "GAIN":
        white = np.array([1.0, 1.0, 1.0])
        white_output = halation(white, white, exposure_lost_lin, green_exposure_lost_lin, blue_exposure_lost_lin)
        output_color = halation_color * white / white_output
    else:
        output_color = halation_color

    return output_color

def main(input_image_path, output_image_path):
    # Load image using rawpy for .RW2 files
    with rawpy.imread(input_image_path) as raw:
        rgb_image = raw.postprocess()

    # Convert the rawpy output to OpenCV format
    image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    height, width, _ = image.shape
    p_TexR, p_TexG, p_TexB = image[:, :, 2], image[:, :, 1], image[:, :, 0]

    output_image = np.zeros_like(image)
    count=0
    print(height*width)
    tic = time.perf_counter()
    for y in range(height):
        for x in range(width):

            count=count+1
            transformed_color = transform(width, height, x, y, p_TexR, p_TexG, p_TexB)
            output_image[y, x] = transformed_color.clip(0, 255).astype(np.uint8)

            if(count==1570):
                toc = time.perf_counter()
                print(f" {toc - tic:0.4f} seconds")
                break
        else:
            continue  # only executed if the inner loop did NOT break
        break  # only executed if the inner loop DID break


    # Save the processed image
    cv2.imwrite(output_image_path, output_image)


input_image_path = 'raw.RW2'
output_image_path = 'output_image.jpg'
main(input_image_path, output_image_path)
