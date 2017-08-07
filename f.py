import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

import pycuda.gpuarray as gpuarray

import pycuda.driver as drv





mod = SourceModule("""
    __global__ void f(int *matrix, int v, int k)
    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     int j = threadIdx.y + blockIdx.y * blockDim.y;

     matrix[i + v * j] = fminf(matrix[i + v * j], matrix[i + v * k] + matrix[k + v * j]);

    }
    """)

m = mod.get_function("f")
                    


mod1 = SourceModule("""
    __global__ void f(int *matrix, int v, int k)
    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     int j = threadIdx.y + blockIdx.y * blockDim.y;

     __shared__ int row[4096];
     __shared__ int col[4096];


     row[i] = matrix[i + v * k];
     col[j] = matrix[k + v * j];

     int temp;
     temp = fminf(matrix[i + v * j], row[i] + col[j]);
     matrix[i + v * j] = temp;

    }
    """)

m1 = mod1.get_function("f")











def fw(m):
    """
    Sequential Floyd-Warshall algorithm.
    
    @param m: numpy array
    @rtype: numpy array

    """
    for k in range(len(m)):
        for i in range(len(m)):
            for j in range(len(m)):
                m[j][i] = min(m[j][i], m[k][i] + m[j][k])

    return m
                




num_of_vertex = 4096
v_numpy = numpy.int32(num_of_vertex)


matrix = numpy.random.randint(1, 99, size=(num_of_vertex, num_of_vertex))
matrix = matrix.astype(numpy.int32)

matrix_gpu = cuda.mem_alloc(matrix.nbytes)
matrix_pin = drv.register_host_memory(matrix)
drv.memcpy_htod_async(matrix_gpu, matrix_pin)

t = time.time()
for k in range(num_of_vertex):
    k_numpy = numpy.int32(k)
    m1(matrix_gpu, v_numpy, k_numpy, grid=(num_of_vertex/32,num_of_vertex/32,1), block=(32,32,1))

print time.time() - t    

result = numpy.empty_like(matrix)
cuda.memcpy_dtoh(result, matrix_gpu)


