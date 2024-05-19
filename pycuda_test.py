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
#include <math.h>
extern "C" {
    __global__ void func(float *pois_lambda, float *uniform_out,
    int *pois_rand, int N,
    float *x_gaussian, float *y_gaussian, float ag, int n_monte_carlo
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        int pix = 0;
        int pt_covered = 0;
        
        float sigmaFilter = 0.8;
        float maxRadius = 0.025;
        
        for (int i=0; i<n_monte_carlo;++i){
            float xGaussian = idx + sigmaFilter*(x_gaussian[i])/ag;
            float yGaussian = idy + sigmaFilter*(y_gaussian[i])/ag;
            
            //determine the bounding boxes around the current shifted pixel
            int minX = floor( (xGaussian - maxRadius)/ag);
            int maxX = floor( (xGaussian + maxRadius)/ag);
            int minY = floor( (yGaussian - maxRadius)/ag);
            int maxY = floor( (yGaussian + maxRadius)/ag);
            
            for(int ncx = minX; ncx <= maxX; ncx++)
            {
                if (pt_covered)
                    break;
                for(int ncy = minY; ncy <= maxY; ncy++)
                {
                    if (pt_covered)
                        break;
                    unsigned long seed = 120;
                    curandState local_state;
                    curand_init(seed, ncx, ncx, &local_state);
                    
                    float u = pois_lambda[idy*N+idx];
                    int ncell = curand_poisson(&local_state, u);
                    for (int k = 0; k < ncell; k++)
                    {
                        float xCentreGrain = ncx*ag + ag*curand_uniform(&local_state);
                        float yCentreGrain = ncy*ag + ag*curand_uniform(&local_state);
                        
                        if ((xCentreGrain-xGaussian)*(xCentreGrain-xGaussian) + (yCentreGrain-yGaussian)*(yCentreGrain-yGaussian) < maxRadius * maxRadius)
                        {
                            ++pix;
                            pt_covered = 1;
                            break;
                        }
                    }
                }
            }
            pt_covered = 0;
        }
        pois_rand[idy*N+idx]=pix;
    }
}
""", no_extern_c=True)
# todo remove simplification of fetching pixel intensity

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

    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo).astype(np.float32)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo).astype(np.float32)

    # Allocate memory on gpu
    img = img_exp[:,:,0].astype(np.float32)
    lambdas_gpu = gpuarray.to_gpu(img)
    x_gaussian_list_gpu = gpuarray.to_gpu(x_gaussian_list)
    y_gaussian_list_gpu = gpuarray.to_gpu(y_gaussian_list)
    sample_gpu_holder = gpuarray.empty((height_in, width_in), dtype=np.int32)
    uniform_gpu_holder = gpuarray.empty((height_in, width_in), dtype=np.float32)

    func(
        lambdas_gpu,
        uniform_gpu_holder,
        sample_gpu_holder,
        np.int32(width_in),
        x_gaussian_list_gpu,
        y_gaussian_list_gpu,
        np.float32(ag),
        np.int32(n_monte_carlo),
        block=(16, 16, 1),
        grid=(16, 16),
        )

    # Retrieve memory from GPU
    sample_gpu_returned = sample_gpu_holder.get()
    plt.imshow(sample_gpu_returned)
    plt.savefig("test.png")

