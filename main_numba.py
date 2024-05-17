import random
import time
import math
import numpy as np
from PIL import Image
from multiprocessing import Pool
from numba import jit

MAX_CHANNELS = 3
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

# arguments of the algorithm
file_name_in = "data/digital/test_image.jpeg"
file_name_out = "image_test_out.jpeg"

mu_r = 0.025
sigma_r = 0.0
sigma_filter = 0.8
n_monte_carlo = 5
zoom = 1.0
grain_seed = random.randint(0, 1000)

height_in = 0
width_in = 0
height_out = 0
width_out = 0


# Square distance
def sq_distance(x1, y1, x2, y2):
    p = [x1, y1]
    q = [x2, y2]
    return math.dist(p, q) ** 2


def cell_seed(x, y, offset):
    period = 2 ** 16
    seed = ((y % period) * period + (x % period)) + offset
    if seed == 0:
        return 1
    return seed % (2 ** 32)


# Render one pixel in the pixel-wise algorithm
def render_pixel(chan_in, chan_out, y_out, x_out, lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list):
#    chan_in, chan_out, y_out, x_out, lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list = args

    normal_quantile = 3.0902
    max_radius = mu_r

    ag = 1 / math.ceil(1 / mu_r)
    zoom_x = (width_out - 1) / (width_in - 1)
    zoom_y = (height_out - 1) / (height_in - 1)

    pix_out = 0.0

    x_in = (x_out + 0.5) * (width_in / width_out)
    y_in = (y_out + 0.5) * (height_in / height_out)

    mu = 0.0
    sigma = 0.0
    if sigma_r > 0.0:
        sigma = math.sqrt(math.log((sigma_r / mu_r) ** 2 + 1.0))
        mu = math.log(mu_r) - sigma ** 2 / 2.0
        log_normal_quantile = math.exp(mu + sigma * normal_quantile)
        max_radius = log_normal_quantile

    for i in range(n_monte_carlo):
        x_gaussian = x_in + sigma_filter * x_gaussian_list[i] / zoom_x
        y_gaussian = y_in + sigma_filter * y_gaussian_list[i] / zoom_y

        min_x = math.floor((x_gaussian - max_radius) / ag)
        max_x = math.floor((x_gaussian + max_radius) / ag)
        min_y = math.floor((y_gaussian - max_radius) / ag)
        max_y = math.floor((y_gaussian + max_radius) / ag)

        pt_covered = False
        for ncx in range(min_x, max_x + 1):
            if pt_covered:
                break
            for ncy in range(min_y, max_y + 1):
                if pt_covered:
                    break
                cell_corner_x = ag * ncx
                cell_corner_y = ag * ncy

                seed = cell_seed(ncx, ncy, grain_seed)
                np.random.seed(seed)

                u = chan_in[min(max(math.floor(cell_corner_y), 0), height_in - 1)][min(max(
                    math.floor(cell_corner_x), 0), width_in - 1)]
                u_ind = int(math.floor(u * MAX_GREY_LEVEL))
                curr_lambda = lambda_list[u_ind]
                curr_exp_lambda = exp_lambda_list[u_ind]

                n_cell = np.random.poisson(curr_lambda * curr_exp_lambda)
                for k in range(n_cell):
                    x_centre_grain = cell_corner_x + ag * np.random.uniform()
                    y_centre_grain = cell_corner_y + ag * np.random.uniform()

                    if sigma_r > 0.0:
                        curr_radius = min(math.exp(mu + sigma * np.random.normal()), max_radius)
                        curr_grain_radius_sq = curr_radius ** 2
                    elif sigma_r == 0.0:
                        curr_grain_radius_sq = mu_r ** 2
                    else:
                        print("Error, the standard deviation of the grain should be positive.")
                        return

                    if sq_distance(x_centre_grain, y_centre_grain, x_gaussian, y_gaussian) < curr_grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        break

    return y_out, x_out, pix_out / n_monte_carlo


# Pixel-wise film grain rendering algorithm
@jit
def film_grain_rendering_pixel_wise(chan_in):
    np.random.seed()

    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)

    lambda_list = np.zeros(MAX_GREY_LEVEL)
    exp_lambda_list = np.zeros(MAX_GREY_LEVEL)
    for i in range(MAX_GREY_LEVEL):
        u = i / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        ag = 1 / math.ceil(1 / mu_r)
        lambda_temp = -(ag ** 2 / (math.pi * (mu_r ** 2 + sigma_r ** 2))) * math.log(1.0 - u)
        lambda_list[i] = lambda_temp
        exp_lambda_list[i] = math.exp(-lambda_temp)

    chan_out = np.zeros((height_out, width_out))

    # Prepare arguments for multiprocessing
    args_list = [(chan_in, chan_out, i, j, lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list)
                 for i in range(img_out.shape[0]) for j in range(img_out.shape[1])]

    # Create a multiprocessing pool
#    with Pool() as pool:
#        results = pool.map(render_pixel, args_list)

#    for i, j, value in results:
#        chan_out[i][j] = value

    for i in range(height_out):
        for j in range(width_out):
            _, _, chan_out[i,j] = render_pixel(chan_in, chan_out, i, j, lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list)

    return chan_out


if __name__ == '__main__':
    image_in = Image.open(file_name_in)
    img_in = np.asarray(image_in)
    img_in = img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)  # normalize the image array

    zoom = 1.0

    width_in = image_in.width
    height_in = image_in.height
    height_out = int(zoom * height_in)
    width_out = int(zoom * width_in)

    # Time and carry out grain rendering
    start = time.time()

    img_out = np.zeros((height_out, width_out, MAX_CHANNELS), dtype=np.uint8)

    for colourChannel in range(MAX_CHANNELS):
        print("_____________________")
        print("Starting colour channel", colourChannel)
        print("_____________________")
        img_in_temp = img_in[:, :, colourChannel]

        # Carry out film grain synthesis
        np.random.seed(1)
        grain_seed = np.random.randint(0, 2**31 - 1)

        img_out_temp = film_grain_rendering_pixel_wise(img_in_temp)
        img_out_temp *= (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        img_out[:, :, colourChannel] = img_out_temp

    image_out = Image.fromarray(img_out)
    image_out.save(file_name_out)

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)