"""
Implementation of some all-pairs shortest path algorithms.
"""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv


"""
Content

1. Sequential implementaion of Floyd-Warshall algorithm
2. Basic parallel implementaion of Floyd-Warshall algorithm
3. Basic parallel implementation of matrix multiplication algorithm
4. Improved parallel implementation of matrix multiplication algorithm
5. Main function

"""










"""
The input is the adjacency matrix of a graph.
"""

#Sequential implementaion of Floyd-Warshall algorithm
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


#Basic parallel implementaion of Floyd-Warshall algorithm
mod = SourceModule("""
    __global__ void f(int *matrix, int v, int k)
    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;
     int j = threadIdx.y + blockIdx.y * blockDim.y;

     matrix[i + v * j] = fminf(matrix[i + v * j], matrix[i + v * k] + matrix[k + v * j]);

    }
    """)

m = mod.get_function("f")

#Basic parallel implementation of matrix multiplication algorithm
mult = SourceModule("""
     __global__ void f(int *matrix, int v)
     {
      int i = threadIdx.x + blockDim.x * blockIdx.x;
      int j = threadIdx.y + blockDim.y * blockIdx.y;

      int tmp;
      tmp = matrix[i + v * j];
      
      for (int k=0;k<v;k++)
      {
       tmp = fminf(tmp, matrix[i + v * k] + matrix[k + v * j]);
      }
      matrix[i + v * j] = tmp;
     }
     """)

mul = mult.get_function("f")

#Improved parallel implementation of matrix multiplication algorith,
mod = SourceModule("""
    __global__ void f(int *m, int v)
    {
     __shared__ int a[32 * 32];
     __shared__ int b[32 * 32];

     int bx = blockIdx.x;
     int by = blockIdx.y;
     int tx = threadIdx.x;
     int ty = threadIdx.y;

     int row = by * 32 + ty;
     int col = bx * 32 + tx;

     int tmp = m[row * v + col];

     for (int l = 0; l<v/32; ++l)
     {
      a[ty + 32 * tx] = m[row * v + l * 32 + tx];
      b[ty + 32 * tx] = m[col + (l * 32 + ty) * v];
      __syncthreads();

      for (int k = 0; k < 32; ++k)
      {
       tmp = fminf(tmp, a[ty + 32 * k] + b[k + tx * v]);
       __syncthreads();
      }
     }

     m[row * v + col] = tmp;

    }
    """)


m = mod.get_function("f")



#Main function

#Declare the source vertex and number of vertex in the graph.
num_of_vertex = 4096
v_numpy = numpy.int32(num_of_vertex)


#Create the adjacency matrix and transfer it to the device
matrix = numpy.random.randint(1, 99, size=(num_of_vertex, num_of_vertex))
matrix = matrix.astype(numpy.int32)

matrix_gpu = cuda.mem_alloc(matrix.nbytes)
matrix_pin = drv.register_host_memory(matrix)
drv.memcpy_htod_async(matrix_gpu, matrix_pin)

#Update the matrix.
#|V| iterations for Floyd-Warshall algorithm
#log|V| iterations for Floyd-Warshall algorithm
for i in range(int(numpy.log2(num_of_vertex))):
    m(matrix_gpu, v_numpy, grid=(num_of_vertex/32,num_of_vertex/32,1), block=(32,32,1))

#Copy the result back to host
result = numpy.empty_like(matrix)
cuda.memcpy_dtoh(result, matrix_gpu)
