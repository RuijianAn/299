 
"""
Implementation of sequential and parallel Bellman-Ford algorithm
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
The graph G = (V, E) is represented by a matrix A.
The vertex in the graph is numbered from 0,1,..., |V|-1, and
A[i][j] represents the weight of the edge between vertex i and vertex j.

In the Bellman-Ford algorithm, each vertex has an attribute 'd' to record
the distance from itself to the source vertex at the moment.
I use an array 'dist' to record it where dist[i] represents the attribute 'd' of vertex i."
"""


#Sequentail Bellman-Ford algorithm

def bellman_ford(dist, matrix, idx):
    """
    Run the bellman-ford algorithm in serial.

    @param dist: list
    @param matrix: list[list]
    @param idx: int
    @rtype: list
    """
    l = len(dist)

    #Initialization
    for i in range(l):
        dist[i] = float('inf')
        if i == idx:
            dist[i] = 0

    #Relaxation
    for j in range(l - 1):
        for k in range(l):
            for m in range(l):
                if m != k and dist[m] + matrix[k][m] < dist[k]:
                    dist[k] = dist[m] + matrix[k][m]
                    
    #Return the shortest path distance
    return dist



#Parallel implementation of Bellman-Ford algorithm

#Initialization
"""
Launch a 1D kernel to initialize mask, mask1, dist.
"""

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


#Relaxation
#Version 1 frontier propagtaion
"""
Map threads to edges.
Check whether the starting vertex is active before relaxation.
Mark the updated vertices.
"""

relax = SourceModule("""
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
         counter[0] = 1;
         }
        }
      }
      """)

relaxation = relax.get_function("relaxa")

#Version 2.1 Multiple edges per thread
"""
Based on Version 1, each thread uses a loop to process multiple edges.
This version's memory access is not coalesced.
"""
relax_multi = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        
        int q;
        int p;
        for (q = 4 * i; q < 4 * i + 4;q++)
        {
         for (p = 4 * j; p < 4 * j + 4;p++)
         {
          if (mask[q] == 1)
          {
           float temp;
           temp = atomicMin(&dist[p], dist[q] + matrix[q + v * p]);
           if (dist[p] < temp)
           {
            mask1[p] = 1;
            counter[0] = 1;
           }
          }
         }
        }
        
      }
      """)

relaxation6 = relax_multi.get_function("relaxa")

#Version 2.2 Multiple edges per thread with coalesced memory access
"""
Adjust the structure of the loop in Version 2.1 to get coalesced memory access.
"""
relax_multi1 = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        
        int q;
        int p;
        for (q = i; q < v;q = q+v/128)
        {
         for (p = j; p < v;p = p+v/128)
         {
          if (mask[q] == 1)
          {
           float temp;
           temp = atomicMin(&dist[p], dist[q] + matrix[q + v * p]);
           if (dist[p] < temp)
           {
            mask1[p] = 1;
            counter[0] = 1;
           }
          }
         }
        }
        
      }
      """)

relaxation7 = relax_multi1.get_function("relaxa")

#Version 3.1 for sparse/general graphs
"""
In this version, 0 in the input matrix represents 2 vertices is not connected in the graph.
Before relaxation, check whether there is an edge between 2 vertices.
However, the adjacency matrix is not an efficient way to represent sparse graphs.
"""
relax_sparse = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        float temp;

        if (mask[i] == 1)
        {
         if (matrix[i + v * j] != 0)
         {
          temp = atomicMin(&dist[j], dist[i] + matrix[i + v * j]);
          if (dist[j] < temp)
          {
          mask1[j] = 1;
          counter[0] = 1;
          }
         }
        }
      }
      """)

relaxation5 = relax_sparse.get_function("relaxa")

#Version 3.2 for sparse /general graphs using an efficient representation
"""
To reduce the required memory, using three arrays s, e, w to store edges,
(s[i], e[i]) is an edge with weight w[i] in the graph.
"""
relax_general = SourceModule("""
     __global__ void re(int *dist, int *s, int *e, int *w, int v, int *mask, int *mask1, int *counter)
     {
      int i = threadIdx.x + blockIdx.x * blockDim.x;

      int temp;
      
      if (mask[s[i]] == 1)
      {
       temp = atomicMin(&dist[e[i]], dist[s[i]] + w[i]);
       if (dist[e[i]] < temp)
       {
        mask1[e[i]] = 1;
        counter[0] = 1;
       }
      }
     }
     """)

relaxation4 = relax_general.get_function("re")

#Version 4.1.1 Incoming edges using a loop
"""
This version maps threads to edges and then relax the incoming edges using a loop.
"""
relax_incoming = SourceModule("""
       __global__ void re(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
       {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        float minimum;
        minimum = dist[idx];
        int i;
        for (i=0;i<v;i++)
        {
         if (mask[i] == 1)
         {
          if (dist[i] + matrix[i + v * idx] < minimum)
          {
           minimum = dist[i] + matrix[i + v * idx];
           mask1[idx] = 1;
           counter[0] = 1;
          }
         }
         dist[idx] = minimum;
        }

       }
       """)
relaxation2 = relax_incoming.get_function("re")

#Version 4.1.2 Incoming edges using a child kernel
"""
Lauch a child kernel which uses reduction to find the minimum.
"""
relaxreduction_cu = '''

__global__ void redu(float *dist, float *matrix, int v, int id)
   {
          int idx = threadIdx.x;
          
          __shared__ float value[512];
          
          value[idx] = dist[idx] + matrix[idx + v * id];
          
          if (idx == id){
             value[idx] = dist[id];
          }
          __syncthreads();

          int s;
          for (s=v/2; s>0;s>>=1){
            if (idx < s){
               if (value[idx] > value[idx + s]){
                  value[idx] = value[idx + s];
               }
            }
          }
          __syncthreads();
          
          if (threadIdx.x == 0)
          {
           dist[id] = value[0];

          }

   }

__global__ void relaxa(float *dist, float *matrix, int v)
      {
       int idx = threadIdx.x;
       redu<<<1, v>>>(dist, matrix, v, idx);
     
      }

'''
mod = DynamicSourceModule(relaxreduction_cu)

relax = mod.get_function("relaxa")

#Swap
"""
Swap mask with mask1.
"""
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

#Main function

#Declare the source vertex and number of vertex in the graph.
num_of_vertex = 4096
v_numpy = numpy.int32(num_of_vertex)
src_idx = numpy.int32(0)

#Create the adjacency matrix, dist, mask, mask1.
matrix = numpy.random.randint(1, 99, size=(num_of_vertex, num_of_vertex))
matrix = matrix.astype(numpy.int32)

dist = numpy.random.randint(1, 99, size=num_of_vertex)
dist = dist.astype(numpy.int32)

mask = numpy.random.randint(1, 99, size=num_of_vertex)
mask = mask.astype(numpy.int32)

mask1 = numpy.random.randint(1, 99, size=num_of_vertex)
mask1 = mask1.astype(numpy.int32)


#Store the matrix in the pinned memory, and then transfer it to the device
matrix_gpu = cuda.mem_alloc(matrix.nbytes)
matrix_pin = drv.register_host_memory(matrix)
drv.memcpy_htod_async(matrix_gpu, matrix_pin)

#Transfer other data to the device in a common way.
dist_gpu = cuda.mem_alloc(dist.nbytes)
dist_pin = drv.register_host_memory(dist)
cuda.memcpy_htod(dist_gpu, dist_pin)

mask_gpu = cuda.mem_alloc(mask.nbytes)
mask_pin = drv.register_host_memory(mask)
cuda.memcpy_htod(mask_gpu, mask_pin)

mask1_gpu = cuda.mem_alloc(mask1.nbytes)
mask1_pin = drv.register_host_memory(mask1)
cuda.memcpy_htod(mask1_gpu, mask1_pin)

#Initialize the dist_gpu.
initilization(dist_gpu, src_idx, mask_gpu, v_numpy, mask1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))

#Version 1 Relax each edge for |V| - 1 times.
for i in range(num_of_vertex - 1):

    #For different versions of relaxation, the kernel call need to be adjusted.
    relaxation3(dist_gpu, matrix_gpu, v_numpy, mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(32,32,1))

    #Copy the flag back to the host
    temp = numpy.empty_like(counter)
    cuda.memcpy_dtoh(temp, counter_gpu)
    #If no vertex is updated, then break out of the loop.
    if temp[0] == 0:
        break
    #Swap mask and mask1
    swap(mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))

#Version 2 Relax each edge for |V| - 1 times using streams.


#Create two streams
s1 = drv.Stream()
s2 = drv.Stream()

#Transfer m1 to the device in stream 1
drv.memcpy_htod_async(m1_gpu, m1_pin, s1)
#Transfer m2 to the device in stream 2
drv.memcpy_htod_async(m2_gpu, m2_pin, s2)

#Relax edges asynchronously in stream1 and stream 2.
for i in range(num_of_vertex - 1):
    relaxation1(dist_gpu, m1_gpu, v_numpy, mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(32,16,1), stream=s1)
    temp = numpy.empty_like(counter)
    drv.memcpy_dtoh_async(temp, counter_gpu,s1)
    if temp[0] == 0:
        break
    swap(mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1), stream=s1)
    


for i in range(num_of_vertex - 1):
    relaxation(dist_gpu, m2_gpu, v_numpy, mask_gpu, mask1_gpu, counter1_gpu, grid=(num_of_vertex /32, num_of_vertex/32, 1), block=(32,16,1), stream=s2)
    temp1 = numpy.empty_like(counter1)
    drv.memcpy_dtoh_async(temp1, counter1_gpu,s2)
    if temp1[0] == 0:
        break
    swap(mask_gpu, mask1_gpu, counter1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1),stream=s2)

#Copy the result back the host.
dist_result = numpy.empty_like(dist)
cuda.memcpy_dtoh(dist_result, dist_gpu)
