import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import pycuda.autoinit
import numpy as np
import os
import matplotlib.pyplot as plt

from pycuda import curandom

if (os.system("cl.exe")):
    os.environ[
        'PATH'] += ';' + r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
if (os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")


# Sets the seed, for reproducibility
def set_seeder(N, seed):
    seedarr = (
            gpuarray.ones_like(gpuarray.zeros(N, dtype=np.int32), dtype=np.int32) * seed
    )
    return seedarr


from pycuda.compiler import SourceModule

func_mod = SourceModule("""
#include <curand_kernel.h>
extern "C" {
    __global__ void func(float pois_lambda, 
    int *pois_rand, 
    curandState *global_state)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curandState local_state = global_state[idx];
        pois_rand[idx] = curand_poisson(&local_state, pois_lambda);
        global_state[idx] = local_state;
    }
}
""", no_extern_c=1)

func = func_mod.get_function('func')

# Define constants
seed = 6
size = 1024
pois_lambda = 10.7
block_size = 64

# Allocate memory on gpu
sample_gpu_holder = gpuarray.empty(size, dtype=np.int32)
# Define the random number generator
_generator = curandom.XORWOWRandomNumberGenerator(
    lambda N: set_seeder(N, seed)
)
max_size = _generator.generators_per_block
if size // max_size > 1:
    raise ValueError("Too many generators expected")

func(
        np.float32(pois_lambda),
        sample_gpu_holder,
        _generator.state,
        block=(block_size, 1, 1),
        grid=(size // block_size + 1, 1),
    )

# Retrieve memory from GPU
sample_gpu_returned = sample_gpu_holder.get()
sample_cpu = np.random.poisson(pois_lambda, size)
hbins = np.arange(20)
plt.hist(sample_gpu_returned, label="GPU samples", alpha=0.5, bins=hbins)
plt.hist(sample_cpu, label="CPU samples", alpha=0.5, bins=hbins)
plt.legend()
plt.savefig("gpu_poisson.png")
