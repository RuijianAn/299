import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import time
import numpy


#numpy.set_printoptions(threshold=numpy.inf)
"""
The graph G = (V, E) is represented by a matrix A.
The vertex in the graph is numbered from 0,1,..., |V|-1, and
A[i][j] represents the weight of the edge between vertex i and vertex j.

In the Bellman-Ford algorithm, each vertex has an attribute 'd' to record
the distance from itself to the source vertex at the moment.
I use an array 'dist' to record it where dist[i] represents the attribute 'd' of vertex i."
"""

#dist, matrix, flag, mask

#Initilization:
#Set the attribute 'd' of all the vertices except source vertex to be INFINITY.
#Set the attribute 'd' of source index to be 0.
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



#Relax:
#For all vertices in the graph, update the attribute 'd'.
#For vertex j, replace the value of dist[j] with min(dist[j], dist[i] + A[i][j])
#where i is the vertex that is adjacent to j and minimize the dist[i] + A[i][j].
relax = SourceModule("""
      __global__ void relaxa(int *dist, int *matrix, int v, int *mask, int *mask1, int *counter)
      {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        float temp;
        temp = dist[j];

        
        if (mask[i] == 1)
        {
         atomicMin(&dist[j], dist[i] + matrix[i + v * j]);
        }
        
        if (dist[j] < temp)
        {
         mask1[j] = 1;
         atomicAdd(&counter[0], 1);
        }
    
      }
      """)

relaxation = relax.get_function("relaxa")

mod = SourceModule("""
    __global__ void swa(int *mask, int *mask1, int *counter)
    {
     int i = threadIdx.x + blockIdx.x * blockDim.x;

     int temp[10240];
     temp[i] = mask1[i];
     __syncthreads();
     
     mask[i] = temp[i];

     mask1[i] = 0;

     if (i == 0)
     {
      counter[i] = 0;
     }

    }
    """)


swap = mod.get_function("swa")



#The Bellman-Ford algorithm in serial.
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

    return dist


#The number of vertices in the graph.
num_of_vertex = 1024

#The number of vertices in the graph which can be passed into kernel.
v_numpy = numpy.int32(num_of_vertex)

#Create a dist array.
dist = numpy.random.randint(1, 99, size=num_of_vertex)
dist = dist.astype(numpy.int32)
dist_gpu = cuda.mem_alloc(dist.nbytes)
cuda.memcpy_htod(dist_gpu, dist)


mask = numpy.random.randint(1, 99, size=num_of_vertex)
mask = mask.astype(numpy.int32)
mask_gpu = cuda.mem_alloc(mask.nbytes)
cuda.memcpy_htod(mask_gpu, mask)

mask1 = numpy.random.randint(1, 99, size=num_of_vertex)
mask1 = mask1.astype(numpy.int32)
mask1_gpu = cuda.mem_alloc(mask1.nbytes)
cuda.memcpy_htod(mask1_gpu, mask1)

#Transfer the dist to GPU.



#Create a random matrix.
#matrix = numpy.random.randint(1, 99, size=(num_of_vertex, num_of_vertex))
matrix = numpy.load('1024.npy')
matrix = matrix.astype(numpy.int32)


#Transfer the matrix to GPU.
matrix_gpu = cuda.mem_alloc(matrix.nbytes)
cuda.memcpy_htod(matrix_gpu, matrix)

#Declare the index of the source vertex.
src_idx = numpy.int32(0)

counter = numpy.array([0])
counter_gpu = cuda.mem_alloc(counter.nbytes)
cuda.memcpy_htod(counter_gpu, counter)

#Record the start time for GPU.
start_time_gpu = time.time()
#Initialize the dist_gpu.
initilization(dist_gpu, src_idx, mask_gpu, v_numpy, mask1_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))

#Relax each edge for |V| - 1 times.
for i in range(num_of_vertex - 1):
    
    
    relaxation(dist_gpu, matrix_gpu, v_numpy, mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/32, num_of_vertex/32, 1), block=(32,32,1))
    temp = numpy.empty_like(counter)
    cuda.memcpy_dtoh(temp, counter_gpu)
    if temp[0] == 0:
        break
    swap(mask_gpu, mask1_gpu, counter_gpu, grid=(num_of_vertex/256,1,1), block=(256,1,1))

counter_result = numpy.empty_like(counter)
cuda.memcpy_dtoh(counter_result, counter_gpu)
print counter_result     
#Copy the dist back to CPU.
dist_result = numpy.empty_like(dist)
cuda.memcpy_dtoh(dist_result, dist_gpu)

#Print the result and running time.
print dist_result
print("--- %s seconds ---" % (time.time() - start_time_gpu))

result = numpy.load('1024o.npy')
print matrix
print result == dist_result
print numpy.array_equal(result, dist_result)



