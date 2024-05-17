import random
import time
import math
import numpy as np
from tqdm import tqdm
from PIL import Image

MAX_CHANNELS = 3
PIXEL_WISE = 0
GRAIN_WISE = 1
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

# arguments of the algorithm
file_name_in = "data/digital/small.png"
file_name_out = "data/small_out.png"


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

    for i in tqdm(range(img_in.shape[0])):
        for j in tqdm(range(img_in.shape[1]), leave=False):
            e = img_in[i, j]
            lambda_val = 1 / (math.pi * (grain_var + grain_radius ** 2)) * math.log(1 / (1 - e))
            model_temp = boolean_model(lambda_val, grain_radius, grain_std)

            for k in range(n_monte_carlo):
                for nGrain in range(model_temp.shape[0]):
                    x_grain_temp = ((model_temp[nGrain, 0] + j) - x_seeds[k] / s_x)[0]
                    y_grain_temp = ((model_temp[nGrain, 1] + i) - y_seeds[k] / s_y)[0]
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
        img_out_temp = film_grain_rendering_grain_wise(img_in_temp, film_grain_params)

        img_out_temp *= (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
        img_out[:, :, colourChannel] = img_out_temp

    image_out = Image.fromarray(img_out)
    image_out.save(file_name_out)

    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)