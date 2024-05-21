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
EPSILON_GREY_LEVEL = 0.01

# arguments of the algorithm
file_name_in = "digital_med.png"
file_name_out = "gpu_med.0.5mu.png"

func_mod = SourceModule("""
#include <curand_kernel.h>
#include <math.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
#define MAX(X, Y) (((X) > (Y)) ? (X) : (Y))

#define BLOCK_SIZE 16

extern "C" {
    __global__ void func(float *pois_lambda, int *pois_rand, int width, int height,
    float *x_gaussian, float *y_gaussian, float ag, int n_monte_carlo, float sigma_filter, float grain_radius,
    int seed
    )
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        
        __shared__ float scratch[BLOCK_SIZE*BLOCK_SIZE];
        if ((idx < width) && (idy < height))
        {
            scratch[threadIdx.y*BLOCK_SIZE + threadIdx.x] = pois_lambda[idy*width + idx];
        }
        
        __syncthreads();
        
        if ((idx < width) && (idy < height))
        {
            int pix = 0;
            int pt_covered = 0;

            
            for (int i=0; i<n_monte_carlo;++i){
                float xGaussian = idx + 0.5 + sigma_filter * x_gaussian[i];
                float yGaussian = idy + 0.5 + sigma_filter * y_gaussian[i];
                
                //determine the bounding boxes around the current shifted pixel
                int minX = floor((xGaussian - grain_radius) / ag);
                int maxX = floor((xGaussian + grain_radius) / ag);
                int minY = floor((yGaussian - grain_radius) / ag);
                int maxY = floor((yGaussian + grain_radius) / ag);
                
                for(int ncx = minX; ncx <= maxX; ncx++)
                {
                    if (pt_covered)
                        break;
                    for(int ncy = minY; ncy <= maxY; ncy++)
                    {
                        if (pt_covered)
                            break;
                        float cell_corner_x = ag * ncx;
                        float cell_corner_y = ag * ncy;
                        curandState local_state;
                        
                        curand_init(seed, ncx, ncy, &local_state);
                        
                        
                        int x_img = MAX(MIN(floor(cell_corner_x),width-1),0);
                        int y_img = MAX(MIN(floor(cell_corner_y),height-1),0);
                        
                        int x_img_pos = x_img - blockIdx.x * blockDim.x;
                        int y_img_pos = y_img - blockIdx.y * blockDim.y;
                        
                        float u;
                        
                        if ((x_img_pos >= 0) && (x_img_pos < BLOCK_SIZE) && (y_img_pos >= 0) && (y_img_pos < BLOCK_SIZE))
                        {
                            u = scratch[threadIdx.y*BLOCK_SIZE + threadIdx.x];
                        } else 
                        {
                            u = pois_lambda[y_img*width + x_img];
                        }

                        int ncell = curand_poisson(&local_state, u);
                        for (int k = 0; k < ncell; k++)
                        {
                            float xCentreGrain = cell_corner_x + ag * curand_uniform(&local_state);
                            float yCentreGrain = cell_corner_y + ag * curand_uniform(&local_state);
                            
                            if ((xCentreGrain-xGaussian)*(xCentreGrain-xGaussian) + (yCentreGrain-yGaussian)*(yCentreGrain-yGaussian) < grain_radius * grain_radius)
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
            pois_rand[idy*width+idx]=pix;
        }    
    }
}
""", no_extern_c=True)

func = func_mod.get_function('func')

if __name__ == '__main__':
    # record time
    start = time.time()

    # Define constants

    image_in = Image.open(file_name_in)
    img_in = np.asarray(image_in)
    # img_in = img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)  # normalize the image array

    width_in = image_in.width
    height_in = image_in.height
    size = (height_in, width_in)

    mu_r = 0.05
    sigma_r = 0.0
    sigma_filter = 0.8
    n_monte_carlo = 5

    ag = 1 / math.ceil(1 / mu_r)
    possible_values = np.arange(MAX_GREY_LEVEL) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)
    lambdas = -(ag ** 2 / (np.pi * (mu_r ** 2 + sigma_r ** 2))) * np.log(1.0 - possible_values)
    lambda_exps = np.exp(-lambdas)

    img_exp = np.take(lambda_exps * lambdas,
                      ((img_in.astype(float) / (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)) * MAX_GREY_LEVEL).astype(int))
    end = time.time()
    print(f"preprocess time {end - start}")

    x_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo).astype(np.float32)
    y_gaussian_list = np.random.normal(0.0, sigma_filter, n_monte_carlo).astype(np.float32)

    # cuda parameters
    block_size = (16, 16, 1)
    grid = (math.ceil(width_in / block_size[1]), math.ceil(height_in / block_size[0]))

    full_img = []

    for color_channel in range(3):
        print("_____________________")
        print("Starting colour channel", color_channel)
        print("_____________________")
        # Allocate memory on gpu
        img = img_exp[:, :, color_channel].astype(np.float32)
        lambdas_gpu = gpuarray.to_gpu(img)
        x_gaussian_list_gpu = gpuarray.to_gpu(x_gaussian_list)
        y_gaussian_list_gpu = gpuarray.to_gpu(y_gaussian_list)
        sample_gpu_holder = gpuarray.empty((height_in, width_in), dtype=np.int32)
        seed = np.uint32(np.random.uniform(0,1000000))

        func(
            lambdas_gpu,
            sample_gpu_holder,
            np.int32(width_in),
            np.int32(height_in),
            x_gaussian_list_gpu,
            y_gaussian_list_gpu,
            np.float32(ag),
            np.int32(n_monte_carlo),
            np.float32(sigma_filter),
            np.float32(mu_r),
            seed,
            block=block_size,
            grid=grid,
        )

        # Retrieve memory from GPU
        full_img.append(sample_gpu_holder.get())

    final_img = ((np.dstack(full_img) / n_monte_carlo) * (MAX_GREY_LEVEL + EPSILON_GREY_LEVEL)).astype(np.uint8)
    image_out = Image.fromarray(final_img)
    image_out.save(file_name_out)

    # record time
    end = time.time()
    elapsed_time = end - start
    print("time elapsed:", elapsed_time)
