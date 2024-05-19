import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import curandom
from PIL import Image
import numpy as np
import time
import os
import math
import matplotlib.pyplot as plt

MAX_CHANNELS = 3
MAX_GREY_LEVEL = 255
EPSILON_GREY_LEVEL = 0.1

# arguments of the algorithm
file_name_in = "data/digital/small.png"
file_name_out = "data/test_small_modified_algo.png"

if (os.system("cl.exe")):
    os.environ[
        'PATH'] += ';' + r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")


func_mod = SourceModule("""
#include <curand_kernel.h>
extern "C" {
    __global__ void func(float *pois_lambda, float *uniform_out,
    int *pois_rand, curandState *global_state, int N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        curandState local_state = global_state[idx];
        pois_rand[idy*N + idx] = curand_poisson(&local_state, pois_lambda[idy*N + idx]);
        uniform_out[idy*N + idx] = curand_uniform(&local_state);
        global_state[idx] = local_state;
    }
}
""", no_extern_c=True)

func = func_mod.get_function('func')

if __name__ == '__main__':
    # Define constants

    image_in = Image.open(file_name_in)
    img_in = np.asarray(image_in)
    # img_in = img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)  # normalize the image array

    width_in = image_in.width
    height_in = image_in.height
    size = (height_in,width_in)

    mu_r = 0.025
    sigma_r = 0.0
    sigma_filter = 0.8
    n_monte_carlo = 100

    ag = 1 / math.ceil(1 / mu_r)
    possible_values = np.arange(MAX_GREY_LEVEL) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
    lambdas = -(ag ** 2 / (np.pi * (mu_r ** 2 + sigma_r ** 2))) * np.log(1.0 - possible_values)
    lambda_exps = np.exp(-lambdas)

    start = time.time()
    img_exp = np.take(lambda_exps * lambdas,
                      ((img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)) * MAX_GREY_LEVEL).astype(int))
    end = time.time()
    print(f"preprocess time {end - start}")


    # Allocate memory on gpu
    img = img_exp[:,:,0].astype(np.float32)
    lambdas_gpu = gpuarray.to_gpu(img)
    sample_gpu_holder = gpuarray.empty((height_in, width_in), dtype=np.int32)
    uniform_gpu_holder = gpuarray.empty((height_in, width_in), dtype=np.float32)
    # Define the random number generator
    _generator = curandom.XORWOWRandomNumberGenerator(
        curandom.seed_getter_unique
    )

    func(
            lambdas_gpu,
            uniform_gpu_holder,
            sample_gpu_holder,
            _generator.state,
            np.int32(width_in),
            block=(16, 16, 1),
            grid=(16, 16),
        )

    # Retrieve memory from GPU
    sample_gpu_returned = sample_gpu_holder.get()
    uniform = uniform_gpu_holder.get()
    print(sample_gpu_returned)
    print(uniform)
    print(_generator.generators_per_block)

