

#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#define NUM_THREADS 4
#define N_V 1024



int flag = 1;
int mask[N_V];

int m[N_V][N_V];
int d[N_V];

int mask1[N_V];






pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t barrier;

void swap(int *a, int *b)
{
    int temp;
    
    temp = *b;
    *b   = *a;
    *a   = temp;
}



void *helper(void *arg){
    int taskid;

    taskid = (int) arg;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j, k;
    for (k=0; k<N_V; k++){
    for (i=first; i<last; i++)
        for (j=0; j<N_V; j++){
            if (mask[j] == 1){
                pthread_mutex_lock(&mutex);
                if (d[i] == INT_MAX && d[i] > d[j] + m[j][i])
                    d[i] = d[j] + m[j][i];
                    mask1[i] = 1;
                    flag = 1;
                pthread_mutex_unlock(&mutex);
            }
            
        }
        pthread_barrier_wait(&barrier);
        if (flag == 0)
            break;
        flag = 0;
        
        for (i=0; i<N_V; i++)
            swap(&mask[i], &mask1[i]);
    }
    free(arg);
    pthread_exit(NULL);
}


int main(){
    int j, p;
    long i;
    clock_t begin = clock();
    for (i=0; i<N_V; i++)
    {
        d[i] = INT_MAX;
        mask[i] = 0;
    }
    
    d[0] = 0;
    mask[0] = 1;
    
    for (p=0; p<N_V; p++)
    {
        for (j=0; j<N_V; j++)
        {
            m[p][j] = rand() % 40;
        }
    }
    
    
    pthread_t thread[NUM_THREADS];
    
    
    flag = 1;
    for (i=0; i<NUM_THREADS; i++){
        pthread_create(&thread[i], NULL, helper, (void *) i);
    }
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;

    
    printf("%f\n", time);
    
    return 0;
    
    
    
    
}


void *helper(void *arg){
    int taskid;
    taskid = (int) arg;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j, k;
    
    for (k=0; k<N_V; k++){
        for (j=0; j<N_V; j++)
        {
         for (i=first; i<last; i++){
            pthread_mutex_lock(&mutex);
            if (d[i] == INT_MAX && d[i] > d[j] + m[j][i])
                d[i] = d[j] + m[j][i];
            pthread_mutex_unlock(&mutex);
         }
        }
        pthread_barrier_wait(&barrier);
    }
    free(arg);
    pthread_exit(NULL);

}



    
int main(){
    int  j, p;
    long i;
    
    
    clock_t begin = clock();
    
    for (i=0; i<N_V; i++)
    {
        d[i] = INT_MAX;
    }
    d[0] = 0;
    
    for (p=0; p<N_V; p++)
    {
        for (j=0; j<N_V; j++)
        {
            m[p][j] = rand() % 40;
        }
    }
    
    
    pthread_t thread[NUM_THREADS];
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    
    for (i=0; i<NUM_THREADS; i++){
        pthread_create(&thread[i], NULL, helper, (void *) i);
    }


    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    printf("%f\n", time);
    
    return 0;
    
}


