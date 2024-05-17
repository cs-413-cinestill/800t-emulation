import random
import time
import math
import numpy as np
from PIL import Image

MAX_CHANNELS = 3
PIXEL_WISE = 0
GRAIN_WISE = 1
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

# arguments of the algorithm
file_name_in = "test_image.jpeg"
file_name_out = "test_image_out.jpeg"

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
def render_pixel(img_in, y_out, x_out, height_in, width_in, height_out, width_out, offset, n_monte_carlo, grain_radius,
                 sigma_r, sigma_filter, lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list):
    normal_quantile = 3.0902
    max_radius = grain_radius

    ag = 1 / math.ceil(1 / grain_radius)
    zoom_x = (width_out - 1) / (width_in - 1)
    zoom_y = (height_out - 1) / (height_in - 1)

    pix_out = 0.0

    x_in = (x_out + 0.5) * (width_in / width_out)
    y_in = (y_out + 0.5) * (height_in / height_out)

    mu = 0.0
    sigma = 0.0
    if sigma_r > 0.0:
        sigma = math.sqrt(math.log((sigma_r / grain_radius) ** 2 + 1.0))
        mu = math.log(grain_radius) - sigma ** 2 / 2.0
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

                seed = cell_seed(ncx, ncy, offset)
                np.random.seed(seed)

                u = img_in[min(max(math.floor(cell_corner_y), 0), height_in - 1)][min(max(
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
                        curr_grain_radius_sq = grain_radius ** 2
                    else:
                        print("Error, the standard deviation of the grain should be positive.")
                        return

                    if sq_distance(x_centre_grain, y_centre_grain, x_gaussian, y_gaussian) < curr_grain_radius_sq:
                        pix_out += 1.0
                        pt_covered = True
                        break
    return pix_out / n_monte_carlo

# Pixel-wise film grain rendering algorithm
def film_grain_rendering_pixel_wise(img_in, film_grain_options):
    grain_radius = film_grain_options["mu_r"]
    grain_std = film_grain_options["sigma_r"]
    sigma_filter = film_grain_options["sigma_filter"]

    n_monte_carlo = film_grain_options["n_monte_carlo"]

    np.random.seed()

    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo)

    lambda_list = np.zeros(MAX_GREY_LEVEL)
    exp_lambda_list = np.zeros(MAX_GREY_LEVEL)
    for i in range(MAX_GREY_LEVEL):
        u = i / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        ag = 1 / math.ceil(1 / grain_radius)
        lambda_temp = -(ag ** 2 / (math.pi * (grain_radius ** 2 + grain_std ** 2))) * math.log(1.0 - u)
        lambda_list[i] = lambda_temp
        exp_lambda_list[i] = math.exp(-lambda_temp)

    img_out = np.zeros((film_grain_options["height_out"], film_grain_options["width_out"]))

    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            print("pixel", i, j)
            print("_________________________")
            pix = render_pixel(img_in, i, j, img_in.shape[0], img_in.shape[1], img_out.shape[0], img_out.shape[1],
                               film_grain_options["grain_seed"], n_monte_carlo, grain_radius, grain_std, sigma_filter,
                               lambda_list, exp_lambda_list, x_gaussian_list, y_gaussian_list)
            img_out[i, j] = pix

    return img_out

# Generate local Boolean model information
def boolean_model(lambda_val, r, std_grain):
    np.random.seed()
    n_dots = np.random.poisson(lambda_val)
    grain_model_out = np.zeros((n_dots, 3))

    for i in range(n_dots):
        grain_model_out[i, 0] = np.random.uniform()
        grain_model_out[i, 1] = np.random.uniform()

    if std_grain == 0.0:
        grain_model_out[:, 2] = r
    else:
        sigma_square = math.log((std_grain / r) ** 2 + 1)
        sigma = math.sqrt(sigma_square)
        mu = math.log(r) - sigma_square / 2.0

        for i in range(n_dots):
            grain_model_out[i, 2] = math.exp(np.random.lognormal(mu, sigma))

    return grain_model_out


# Grain-wise film grain rendering algorithm
def film_grain_rendering_grain_wise(img_in, film_grain_options):
    grain_radius = film_grain_options["mu_r"]
    grain_std = film_grain_options["sigma_r"]
    grain_var = grain_std ** 2
    sigma_filter = film_grain_options["sigma_filter"]
    s_x = film_grain_options["width_out"] / (img_in.shape[1] - 1)
    s_y = film_grain_options["height_out"] / (img_in.shape[0] - 1)

    n_monte_carlo = film_grain_options["n_monte_carlo"]

    np.random.seed()
    x_seeds = np.random.normal(0.0, sigma_filter, (n_monte_carlo, 1))
    y_seeds = np.random.normal(0.0, sigma_filter, (n_monte_carlo, 1))

    img_lambda = np.zeros((img_in.shape[0], img_in.shape[1]))
    img_out = np.zeros((film_grain_options["height_out"], film_grain_options["width_out"]))

    img_temp_ptr = [np.zeros((film_grain_options["height_out"],
                              film_grain_options["width_out"]), dtype=bool) for _ in range(n_monte_carlo)]

    for i in range(img_in.shape[0]):
        for j in range(img_in.shape[1]):
            e = img_in[i, j]
            lambda_val = 1 / (math.pi * (grain_var + grain_radius ** 2)) * math.log(1 / (1 - e))
            model_temp = boolean_model(lambda_val, grain_radius, grain_std)

            for k in range(n_monte_carlo):
                for nGrain in range(model_temp.shape[0]):
                    x_grain_temp = (model_temp[nGrain, 0] + j) - x_seeds[k] / s_x
                    y_grain_temp = (model_temp[nGrain, 1] + i) - y_seeds[k] / s_y
                    r_grain_temp = model_temp[nGrain, 2]
                    r_grain_temp_sq = r_grain_temp ** 2

                    min_b_bx = math.ceil((x_grain_temp - r_grain_temp) * s_x)
                    max_b_bx = math.floor((x_grain_temp + r_grain_temp) * s_x)
                    min_b_by = math.ceil((y_grain_temp - r_grain_temp) * s_y)
                    max_b_by = math.floor((y_grain_temp + r_grain_temp) * s_y)

                    for x in range(min_b_bx, max_b_bx + 1):
                        for y in range(min_b_by, max_b_by + 1):
                            if 0 <= y < img_out.shape[0] and 0 <= x < img_out.shape[1]:
                                if (y / s_y - y_grain_temp) ** 2 + (x / s_x - x_grain_temp) ** 2 <= r_grain_temp_sq:
                                    img_temp_ptr[k][int(y), int(
                                        min(max(x, 0), img_out.shape[1] - 1))] = True

            img_lambda[i, j] = lambda_val
            del model_temp

    for i in range(img_out.shape[0]):
        for j in range(img_out.shape[1]):
            val_temp = sum(img_temp_ptr[k][i, j] for k in range(n_monte_carlo))
            img_out[i, j] = val_temp

    img_out /= n_monte_carlo

    return img_out

if __name__ == '__main__':
    image_in = Image.open(file_name_in)
    img_in = np.asarray(image_in)
    img_in = img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)  # normalize the image array

    zoom = 1.0

    width_in = image_in.width
    height_in = image_in.height
    height_out = int(zoom * height_in)
    width_out = int(zoom * width_in)

    mu_r = 0.025
    sigma_r = 0.0
    sigma_filter = 0.8
    n_monte_carlo = 5
    algorithm_id = 0

    film_grain_params = {
        "mu_r": mu_r,
        "sigma_r": sigma_r,
        "zoom": zoom,
        "sigma_filter": sigma_filter,
        "n_monte_carlo": n_monte_carlo,
        "algorithm_id": algorithm_id,
        "height_out": height_out,
        "width_out": width_out,
        "grain_seed": random.randint(0, 1000)
    }

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
        film_grain_params["grainSeed"] = np.random.randint(0, 2**31 - 1)
        if film_grain_params["algorithm_id"] == PIXEL_WISE:
            img_out_temp = film_grain_rendering_pixel_wise(img_in_temp, film_grain_params)
        else:
            img_out_temp = film_grain_rendering_grain_wise(img_in_temp, film_grain_params)

        img_out_temp *= (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        img_out[:, :, colourChannel] = img_out_temp

    image_out = Image.fromarray(img_out)
    image_out.save(file_name_out)

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)