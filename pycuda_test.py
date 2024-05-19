import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import curandom
import numpy as np
import os
import matplotlib.pyplot as plt


if (os.system("cl.exe")):
    os.environ[
        'PATH'] += ';' + r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")


func_mod = SourceModule("""
#include <curand_kernel.h>
extern "C" {
    __global__ void func(float *pois_lambda, float *uniform_out,
    int *pois_rand, curandState *global_state)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curandState local_state = global_state[idx];
        pois_rand[idx] = curand_poisson(&local_state, pois_lambda[idx]);
        uniform_out[idx] = curand_uniform(&local_state);
        global_state[idx] = local_state;
    }
}
""", no_extern_c=1)

func = func_mod.get_function('func')

if __name__ == '__main__':
    # Define constants
    size = 4
    block_size = 64

    # Allocate memory on gpu
    lambdas = np.array([1,2,3,4], dtype=np.float32)
    lambdas_gpu = gpuarray.to_gpu(lambdas)
    sample_gpu_holder = gpuarray.empty(size, dtype=np.int32)
    uniform_gpu_holder = gpuarray.empty(size, dtype=np.float32)
    # Define the random number generator
    _generator = curandom.XORWOWRandomNumberGenerator(
        curandom.seed_getter_unique
    )

    func(
            lambdas_gpu,
            uniform_gpu_holder,
            sample_gpu_holder,
            _generator.state,
            block=(block_size, 1, 1),
            grid=(size // block_size + 1, 1),
        )

    # Retrieve memory from GPU
    sample_gpu_returned = sample_gpu_holder.get()
    uniform = uniform_gpu_holder.get()
    print(sample_gpu_returned)
    print(uniform)

