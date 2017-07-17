import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy
from pycuda.compiler import DynamicSourceModule
import pycuda.autoinit

import pycuda.gpuarray as gpuarray

import pycuda.driver as drv


numpy.set_printoptions(threshold=numpy.inf)


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

initilization = initi.get_function("ini")


relax = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        float temp;

        if (mask[i] == 1)
        {
         temp = atomicMin(&dist[j + v/2], dist[i] + matrix[i + v * j]);
         if (dist[j + v/2] < temp)
         {
         mask1[j + v/2] = 1;
         counter = 1;
         }
        }
      }
      """)

relaxation = relax.get_function("relaxa")


relax0 = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;

        float temp;

        if (mask[i] == 1)
        {
         temp = atomicMin(&dist[j], dist[i] + matrix[i + v * j]);
         if (dist[j] < temp)
         {
         mask1[j] = 1;
         counter = 1;
         }
        }
      }
      """)

relaxation1 = relax0.get_function("relaxa")


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
      counter = 0;
     }

    }
    """)


swap = mod.get_function("swa")


num_of_vertex = 2048

v_numpy = numpy.int32(num_of_vertex)

src_idx = numpy.int32(0)
#First part of the matirx
m1 = numpy.random.randint(1, 99, size=(num_of_vertex/2, num_of_vertex))

m1 = m1.astype(numpy.int32)
#Second part of the matrix
m2 = numpy.random.randint(1, 99, size=(num_of_vertex/2, num_of_vertex))

m2 = m2.astype(numpy.int32)

counter1 = numpy.int32(0)
counter = numpy.int32(0)

dist = numpy.random.randint(1, 99, size=num_of_vertex)
dist = dist.astype(numpy.int32)

mask = numpy.random.randint(1, 99, size=num_of_vertex)
mask = mask.astype(numpy.int32)

mask1 = numpy.random.randint(1, 99, size=num_of_vertex)
mask1 = mask1.astype(numpy.int32)



t = time.time()


m1_gpu = cuda.mem_alloc(m1.nbytes)
m1_pin = drv.register_host_memory(m1)


m2_gpu = cuda.mem_alloc(m2.nbytes)
m2_pin = drv.register_host_memory(m2)

counter_gpu = cuda.mem_alloc(counter.nbytes)
cuda.memcpy_htod(counter_gpu, counter)

counter1_gpu = cuda.mem_alloc(counter1.nbytes)
cuda.memcpy_htod(counter1_gpu, counter1)

dist_gpu = cuda.mem_alloc(dist.nbytes)
dist_pin = drv.register_host_memory(dist)
cuda.memcpy_htod(dist_gpu, dist_pin)


mask_gpu = cuda.mem_alloc(mask.nbytes)
mask_pin = drv.register_host_memory(mask)
cuda.memcpy_htod(mask_gpu, mask_pin)


mask1_gpu = cuda.mem_alloc(mask1.nbytes)
mask1_pin = drv.register_host_memory(mask1)
cuda.memcpy_htod(mask1_gpu, mask1_pin)


initilization(dist_gpu, src_idx, mask_gpu, v_numpy, mask1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))



#Create two streams
s1 = drv.Stream()
s2 = drv.Stream()

drv.memcpy_htod_async(m1_gpu, m1_pin, s1)
drv.memcpy_htod_async(m2_gpu, m2_pin, s2)
#First process the first part of the matrix 
for i in range(num_of_vertex - 1):
    relaxation1(dist_gpu, m1_gpu, v_numpy, mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(32,16,1), stream=s1)
    temp = numpy.empty_like(counter)
    drv.memcpy_dtoh_async(temp, counter_gpu,s1)
    if temp == 0:
        break
    swap(mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1), stream=s1)
    


for i in range(num_of_vertex - 1):
    relaxation(dist_gpu, m2_gpu, v_numpy, mask_gpu, mask1_gpu, counter1_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(32,16,1), stream=s2)
    temp1 = numpy.empty_like(counter1)
    drv.memcpy_dtoh_async(temp1, counter1_gpu,s2)
    if temp1 == 0:
        break
    swap(mask_gpu, mask1_gpu, counter1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1),stream=s2)

dist_result = numpy.empty_like(dist)
cuda.memcpy_dtoh(dist_result, dist_gpu)


print("--- %s seconds ---" % (time.time() - t))
