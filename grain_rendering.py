import random
import time
import math
import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

MAX_CHANNELS = 3
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

mu_r = 0.1
sigma_r = 0
sigma_filter = 0.8
n_monte_carlo = 5

height_in = 0
width_in = 0

# arguments of the algorithm
file_name_in = "data/digital/after_transfer.png"
file_name_out = "data/digital/after_grain.png"


@njit
def cell_seed(x, y, offset):
    period = 2 ** 16
    seed = ((y % period) * period + (x % period)) + offset
    if seed == 0:
        return 1
    return seed % (2 ** 32)


@njit
def poisson(lambda_val):
    x = np.random.poisson(lambda_val)
    if x > 10000 * lambda_val:
        x = 10000 * lambda_val
    return x


# Render one pixel in the pixel-wise algorithm
@njit
def render_pixel(chan_in, y_out, x_out, offset, x_gaussian_list, y_gaussian_list):
    normal_quantile = 3.0902
    max_radius = mu_r
    ag = 1 / math.ceil(1 / mu_r)

    pix_out = 0.0
    x_in = x_out + 0.5
    y_in = y_out + 0.5

    mu = 0.0
    sigma = 0.0
    if sigma_r > 0.0:
        sigma = math.sqrt(math.log((sigma_r / mu_r) ** 2 + 1.0))
        mu = math.log(mu_r) - sigma ** 2 / 2.0
        log_normal_quantile = math.exp(mu + sigma * normal_quantile)
        max_radius = log_normal_quantile

    for i in range(n_monte_carlo):
        x_gaussian = x_in + sigma_filter * x_gaussian_list[i]
        y_gaussian = y_in + sigma_filter * y_gaussian_list[i]

        min_x = int((x_gaussian - max_radius) / ag)
        max_x = int((x_gaussian + max_radius) / ag)
        min_y = int((y_gaussian - max_radius) / ag)
        max_y = int((y_gaussian + max_radius) / ag)

        pt_covered = False
        for ncx in range(min_x, max_x + 1):
            if pt_covered:
                break
            for ncy in range(min_y, max_y + 1):
                if pt_covered:
                    break
                cell_corner_x = ag * ncx
                cell_corner_y = ag * ncy

                seed = cell_seed(ncx, ncy, offset)
                np.random.seed(seed)

                lambda_val = chan_in[min(max(int(cell_corner_y), 0), height_in - 1)][min(max(
                    int(cell_corner_x), 0), width_in - 1)]

                n_cell = poisson(lambda_val)

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

                    if (x_centre_grain - x_gaussian) ** 2 + (y_centre_grain - y_gaussian) ** 2 < curr_grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        break
    return pix_out / n_monte_carlo


# Pixel-wise film grain rendering algorithm
@njit(parallel=True)
def film_grain_rendering_pixel_wise(chan_in, grain_seed, progress_proxy):
    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)

    chan_out = np.zeros(chan_in.shape)

    for i in prange(chan_out.shape[0]):
        for j in prange(chan_out.shape[1]):
            pix = render_pixel(chan_in, i, j, grain_seed, x_gaussian_list, y_gaussian_list)
            chan_out[i, j] = pix
        progress_proxy.update(1)

    return chan_out


def grain_rendering(img_in):
    width_in = img_in.shape[0]
    height_in = img_in.shape[1]

    ag = 1 / math.ceil(1 / mu_r)
    possible_values = np.arange(MAX_GREY_LEVEL) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
    lambdas = -(ag ** 2 / (np.pi * (mu_r ** 2 + sigma_r ** 2))) * np.log(1.0 - possible_values)

    start = time.time()
    img_lambda = np.take(lambdas, ((img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)) * MAX_GREY_LEVEL).astype(int))
    end = time.time()
    print(f"preprocess time {end-start}")


    print("_____________________")
    print("trigger function compilation")
    print("_____________________")
    img_in_temp = np.zeros((2, 2, 3))[:, :, 0]  # cannot remove slicing or runs much slower at start of color channel 0
    # trigger function compilation
    with ProgressBar(total=2) as progress:
        film_grain_rendering_pixel_wise(img_in_temp, random.randint(0, 1000), progress)

    # Carry out film grain synthesis
    img_out = np.zeros((height_in, width_in, MAX_CHANNELS), dtype=np.uint8)
    # Time and carry out grain rendering
    start = time.time()
    for colourChannel in range(MAX_CHANNELS):
        print("_____________________")
        print("Starting colour channel", colourChannel)
        print("_____________________")
        img_in_temp = img_lambda[:, :, colourChannel]

        # Carry out film grain synthesis
        img_out_temp = []
        with ProgressBar(total=img_in.shape[0]) as progress:
            img_out_temp = film_grain_rendering_pixel_wise(img_in_temp, random.randint(0, 1000), progress)
        img_out_temp *= (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        img_out[:, :, colourChannel] = img_out_temp

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)

    return img_out


def grain_interface(img_in):
    grain_rendering(img_in)
