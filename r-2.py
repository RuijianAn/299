import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

#Initialization
initi = SourceModule("""
  __global__ void ini(int *a, int i, int *mask, int v, int *mask1)
  {
    #include <math.h>
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    a[idx] = INFINITY;
    mask[idx] = 0;
    mask1[idx] = 0;
    if (idx == i)
    {
       a[idx] = 0;
       mask[idx] = 1;
    }
    
  }
  """)

initialization = initi.get_function("ini")


relax_multi1 = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        
        int q;
        int p;
        for (q = i; q < v;q = q+blockDim.x * gridDim.x)
        {
         for (p = j; p < v;p = p+blockDim.y * gridDim.y)
         {
          if (mask[q] == 1)
          {
           float temp;
           temp = atomicMin(&dist[p], dist[q] + matrix[q + v * p]);
           if (dist[p] < temp)
           {
            mask1[p] = 1;
            counter = 1;
           }
          }
         }
        }
        
      }
      """)

relaxation7 = relax_multi1.get_function("relaxa")


mod = SourceModule("""
    __global__ void swa(int *mask, int *mask1, int *counter)
    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;

     int temp[20480];
     temp[i] = mask1[i];
     
     mask[i] = temp[i];

     mask1[i] = 0;

     if (i == 0)
     {
      counter[i] = 0;
     }

    }
    """)


swap = mod.get_function("swa")


#The number of vertices in the graph.
num_of_vertex = 20480
#The number of vertices in the graph which can be passed into kernel.
v_numpy = numpy.int32(num_of_vertex)

#Create a dist array and transfer it to device.
dist = numpy.random.randint(1, 99, size=num_of_vertex)
dist = dist.astype(numpy.int32)
dist_gpu = cuda.mem_alloc(dist.nbytes)
cuda.memcpy_htod(dist_gpu, dist)

#Create mask&mask1 and transfer them to device.
mask = numpy.random.randint(1, 99, size=num_of_vertex)
mask = mask.astype(numpy.int32)
mask_gpu = cuda.mem_alloc(mask.nbytes)
cuda.memcpy_htod(mask_gpu, mask)

mask1 = numpy.random.randint(1, 99, size=num_of_vertex)
mask1 = mask1.astype(numpy.int32)
mask1_gpu = cuda.mem_alloc(mask1.nbytes)
cuda.memcpy_htod(mask1_gpu, mask1)


#Create a random matrix.
matrix = numpy.random.randint(1, 99, size=(num_of_vertex, num_of_vertex))
matrix = matrix.astype(numpy.int32)


#Transfer the matrix to GPU.
matrix_gpu = cuda.mem_alloc(matrix.nbytes)
cuda.memcpy_htod(matrix_gpu, matrix)

#Declare the index of the source vertex.
src_idx = numpy.int32(0)

counter = 0
counter_gpu = cuda.mem_alloc(counter.nbytes)
cuda.memcpy_htod(counter_gpu, counter)

#Record the start time for GPU.
start_time_gpu = time.time()
#Initialize the dist_gpu.
initilization(dist_gpu, src_idx, mask_gpu, v_numpy, mask1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))


#Relax each edge for |V| - 1 times.
for i in range(num_of_vertex - 1):
    
    relaxation7(dist_gpu, matrix_gpu, v_numpy, mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(4,4,1))
    
    temp = numpy.empty_like(counter)
    cuda.memcpy_dtoh(temp, counter_gpu)
    if temp == 0:
        break
    swap(mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))



#Copy the dist back to CPU.
dist_result = numpy.empty_like(dist)
cuda.memcpy_dtoh(dist_result, dist_gpu)

#Print the result and running time.

print("--- %s seconds ---" % (time.time() - start_time_gpu))



