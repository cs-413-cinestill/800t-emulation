import random
import time
import math
import numpy as np
from numba import njit, prange
from numba_progress import ProgressBar


"""
Implements the necessary functions the needed functions to generate and render grain on an image.

The grain can be altered using multiple parameters.

mu_r: float, the average size of the grain, in unit relative to the input image grid
sigma_r: float, the standard deviation of the grain, relative to mu_r
sigma_filter: float, the sigma of the gaussian filter's distribution
n_monte_carlo: int, the number of Monte Carlo iterations to simulate the gaussian distribution

It also uses a few constants.

NUM_CHANNELS: the number of color channels on which to compute
MAX_INTENSITY: the maximum intensity value in an image
EPSILON_INTENSITY: small parameter to avoid some division by 0
"""

mu_r = 0.15
sigma_r = 0
sigma_filter = 0.8
n_monte_carlo = 100

NUM_CHANNELS = 3
MAX_INTENSITY = 255
EPSILON_INTENSITY = 0.1


@njit
def cell_seed(x, y, offset):
    """
    Generate a seed for the poisson distribution for a specific cell

    :param x: x coordinate of the cell
    :param y: y coordinate of the cell
    :param offset: random offset generated for the color channel
    :return: the generated seed
    """
    period = 2 ** 16
    seed = ((y % period) * period + (x % period)) + offset
    if seed == 0:
        return 1
    return seed % (2 ** 32)


@njit
def bounded_poisson(lambda_val):
    """
    Generate a value following a poisson distribution, bounded to avoid outliers from affecting the result
    too much

    :param lambda_val: the lambda of the poisson distribution
    :return: the generated value
    """
    x = np.random.poisson(lambda_val)
    if x > 10000 * lambda_val:
        x = 10000 * lambda_val
    return x


@njit
def render_pixel(chan_in, y_out, x_out, seed_offset, x_gaussian_list, y_gaussian_list, width_in, height_in):
    """
    Compute the grey value for a pixel of the output image in one color channel

    :param chan_in: the color channel of the input image
    :param y_out: the y coordinate of the output pixel
    :param x_out: the x coordinate of the output pixel
    :param seed_offset: random offset used to compute a cell's seed
    :param x_gaussian_list: list of n_monte_carlo values following a gaussian distribution,
        used to offset the input pixel along the x-axis
    :param y_gaussian_list: list of n_monte_carlo values following a gaussian distribution,
        used to offset the input pixel along the y-axis
    :param width_in: width of the input image
    :param height_in: height of the input image
    :return: the normalized grey value of the pixel at this color channel
    """
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
        # offset the input pixel using a gaussian distribution
        x_gaussian = x_in + sigma_filter * x_gaussian_list[i]
        y_gaussian = y_in + sigma_filter * y_gaussian_list[i]

        # compute the cell coordinates to which the offset pixel belongs
        min_x = int((x_gaussian - max_radius) / ag)
        max_x = int((x_gaussian + max_radius) / ag)
        min_y = int((y_gaussian - max_radius) / ag)
        max_y = int((y_gaussian + max_radius) / ag)

        pt_covered = False
        for nc_x in range(min_x, max_x + 1):
            if pt_covered:
                break
            for nc_y in range(min_y, max_y + 1):
                if pt_covered:
                    break
                cell_corner_x = ag * nc_x
                cell_corner_y = ag * nc_y

                seed = cell_seed(nc_x, nc_y, seed_offset)
                np.random.seed(seed)

                lambda_val = chan_in[min(max(int(cell_corner_y), 0), height_in - 1)][min(max(
                    int(cell_corner_x), 0), width_in - 1)]

                n_cell = bounded_poisson(lambda_val)

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
def film_grain_rendering_pixel_wise(chan_in, seed_offset, progress_proxy, width_in, height_in):
    """
    Computes the output image on one color channel

    :param chan_in: the input color channel
    :param seed_offset: a random offset used to compute the seed of the grain amount
    :param progress_proxy: the progress bar of the program execution
    :param width_in: the width of the input image
    :param height_in: the height of the input image
    :return: the color channel of the output image
    """
    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)

    chan_out = np.zeros(chan_in.shape)

    for i in prange(chan_out.shape[0]):
        for j in prange(chan_out.shape[1]):
            pix = render_pixel(chan_in, i, j, seed_offset, x_gaussian_list, y_gaussian_list, width_in, height_in)
            chan_out[i, j] = pix
        progress_proxy.update(1)

    return chan_out


def grain_rendering(img_in):
    """
    Renders the grain on an input image

    :param img_in: the image on which to render grain
    :return: the image with grain rendered
    """
    width_in = img_in.shape[0]
    height_in = img_in.shape[1]

    ag = 1 / math.ceil(1 / mu_r)
    possible_values = np.arange(MAX_INTENSITY) / (MAX_INTENSITY + EPSILON_INTENSITY)
    lambdas = -(ag ** 2 / (np.pi * (mu_r ** 2 + sigma_r ** 2))) * np.log(1.0 - possible_values)

    start = time.time()
    # Pre-compute the lambda of the poisson distribution at every pixel
    img_lambda = np.take(lambdas, ((img_in.astype(float) / (MAX_INTENSITY + EPSILON_INTENSITY)) * MAX_INTENSITY).astype(int))
    end = time.time()
    print(f"preprocess time {end-start}")


    print("_____________________")
    print("trigger function compilation")
    print("_____________________")
    img_in_temp = np.zeros((2, 2, 3))[:, :, 0]  # cannot remove slicing or runs much slower at start of color channel 0
    # trigger function compilation
    with ProgressBar(total=2) as progress:
        film_grain_rendering_pixel_wise(img_in_temp, random.randint(0, 1000), progress, width_in, height_in)

    # Carry out film grain synthesis
    img_out = np.zeros((height_in, width_in, NUM_CHANNELS), dtype=np.uint8)
    # Time and carry out grain rendering
    start = time.time()
    for colourChannel in range(NUM_CHANNELS):
        print("_____________________")
        print("Starting colour channel", colourChannel)
        print("_____________________")
        img_in_temp = img_lambda[:, :, colourChannel]

        # Carry out film grain synthesis
        img_out_temp = []
        with ProgressBar(total=img_in.shape[0]) as progress:
            img_out_temp = film_grain_rendering_pixel_wise(img_in_temp, random.randint(0, 1000), progress, width_in, height_in)
        img_out_temp *= (MAX_INTENSITY + EPSILON_INTENSITY)
        img_out[:, :, colourChannel] = img_out_temp

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)

    return img_out


def grain_interface(img_in):
    """
    Simple interface function used to call the grain rendering algorithm from other files

    :param img_in: the image on which to render grain
    :return: the image with grain rendered
    """
    return grain_rendering(img_in)