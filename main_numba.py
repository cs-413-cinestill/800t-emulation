import random
import time
import math
import numpy as np
from PIL import Image
from numba import njit, prange
from numba_progress import ProgressBar

MAX_CHANNELS = 3
PIXEL_WISE = 0
GRAIN_WISE = 1
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

# arguments of the algorithm
file_name_in = "data/small.png"
file_name_out = "data/test_small_modified_algo.png"


@njit
def cell_seed(x, y, offset):
    period = 2 ** 16
    seed = ((y % period) * period + (x % period)) + offset
    if seed == 0:
        return 1
    return seed % (2 ** 32)


# Render one pixel in the pixel-wise algorithm
@njit
def render_pixel(img_in, y_out, x_out, height_in, width_in, offset, n_monte_carlo, grain_radius,
                 sigma_r, sigma_filter, x_gaussian_list, y_gaussian_list):

    ag = 1 / math.ceil(1 / grain_radius)

    pix_out = 0.0

    for i in range(n_monte_carlo):
        x_gaussian = x_out + 0.5 + sigma_filter * x_gaussian_list[i]
        y_gaussian = y_out + 0.5 + sigma_filter * y_gaussian_list[i]

        min_x = int((x_gaussian - grain_radius) / ag)
        max_x = int((x_gaussian + grain_radius) / ag)
        min_y = int((y_gaussian - grain_radius) / ag)
        max_y = int((y_gaussian + grain_radius) / ag)

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

                n_cell = np.random.poisson(img_in[min(max(int(cell_corner_y), 0), height_in - 1)][min(max(
                    int(cell_corner_x), 0), width_in - 1)])
                for k in range(n_cell):
                    x_centre_grain = cell_corner_x + ag * np.random.uniform()
                    y_centre_grain = cell_corner_y + ag * np.random.uniform()

                    # if sigma_r > 0.0:
                    #     curr_radius = min(math.exp(mu + sigma * np.random.normal()), max_radius)
                    #     curr_grain_radius_sq = curr_radius ** 2
                    # elif sigma_r == 0.0:
                    # else:
                    #     print("Error, the standard deviation of the grain should be positive.")
                    #     return

                    if (x_centre_grain - x_gaussian) ** 2 + (y_centre_grain - y_gaussian) ** 2 < grain_radius ** 2:
                        pix_out += 1.0
                        pt_covered = True
                        break
    return pix_out / n_monte_carlo


# Pixel-wise film grain rendering algorithm
@njit(parallel=True)
def film_grain_rendering_pixel_wise(img_in, grain_radius, grain_std, sigma_filter, n_monte_carlo,
                                    grain_seed, progress_proxy):
    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)

    img_out = np.zeros(img_in.shape)

    for i in prange(img_out.shape[0]):
        for j in prange(img_out.shape[1]):
            pix = render_pixel(img_in, i, j, img_in.shape[0], img_in.shape[1],
                               grain_seed, n_monte_carlo, grain_radius, grain_std, sigma_filter,
                               x_gaussian_list, y_gaussian_list)
            img_out[i, j] = pix
        progress_proxy.update(1)

    return img_out


if __name__ == '__main__':
    image_in = Image.open(file_name_in)
    img_in = np.asarray(image_in)
    # img_in = img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)  # normalize the image array

    width_in = image_in.width
    height_in = image_in.height

    mu_r = 0.025
    sigma_r = 0.0
    sigma_filter = 0.8
    n_monte_carlo = 100

    ag = 1 / math.ceil(1 / mu_r)
    possible_values = np.arange(MAX_GREY_LEVEL) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
    lambdas = -(ag ** 2 / (np.pi * (mu_r ** 2 + sigma_r ** 2))) * np.log(1.0 - possible_values)
    lambda_exps = np.exp(-lambdas)

    start = time.time()
    img_exp = np.take(lambda_exps * lambdas, ((img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL))*MAX_GREY_LEVEL).astype(int))
    end = time.time()
    print(f"preprocess time {end-start}")


    print("_____________________")
    print("trigger function compilation")
    print("_____________________")
    img_in_temp = np.zeros((2, 2, 3))[:, :, 0]  # cannot remove slicing or runs much slower at start of color channel 0
    # trigger function compilation
    with ProgressBar(total=2) as progress:
        film_grain_rendering_pixel_wise(img_in_temp, mu_r, sigma_r, sigma_filter, n_monte_carlo,
                                        random.randint(0, 1000), progress)

    # Carry out film grain synthesis
    img_out = np.zeros((height_in, width_in, MAX_CHANNELS), dtype=np.uint8)
    # Time and carry out grain rendering
    start = time.time()
    for colourChannel in range(MAX_CHANNELS):
        print("_____________________")
        print("Starting colour channel", colourChannel)
        print("_____________________")
        img_in_temp = img_exp[:, :, colourChannel]

        # Carry out film grain synthesis
        img_out_temp = []
        with ProgressBar(total=img_in.shape[0]) as progress:
            img_out_temp = film_grain_rendering_pixel_wise(img_in_temp, mu_r, sigma_r, sigma_filter, n_monte_carlo,
                                                           random.randint(0, 1000), progress)
        img_out_temp *= (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        img_out[:, :, colourChannel] = img_out_temp

    image_out = Image.fromarray(img_out)
    image_out.save(file_name_out)

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)
