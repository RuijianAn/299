

#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#define NUM_THREADS 4
#define N_V 4096


int flag = 1;
int mask[N_V];

int m[N_V][N_V];
int d[N_V];

int mask1[N_V];




pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_barrier_t bar;

void swap(int *a, int *b)
{
    int temp;
    
    temp = *b;
    *b   = *a;
    *a   = temp;
}


void *ini(void *arg){
    long taskid;
    taskid = (int) arg;
    
    long first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    long i;
    for (i=first; i<last; i++)
    {
        d[i] = INT_MAX;
        mask[i] = 0;
    }
    
    pthread_barrier_wait(&bar);
    pthread_exit(NULL);
        
}


void *helper(void *arg){
    long taskid;
    taskid = (int) arg;
    
    long first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    long i, j, k;
    for (k=0; k<N_V; k++)
    {
        for (i=first; i<last; i++)
        {
            for (j=0; j<N_V; j++)
            {
                if (mask[j] == 1)
                {
                    pthread_mutex_lock(&mutex);
                    if (d[i] == INT_MAX || d[i] > d[j] + m[j][i])
                    {
                        d[i] = d[j] + m[j][i];
                        mask1[i] = 1;
                        flag = 1;
                    }
                    pthread_mutex_unlock(&mutex);
                }
            }
        }
        pthread_barrier_wait(&bar);
        if (flag == 0)
            pthread_exit(NULL);
        if (taskid == 0)
        {
            flag = 0;
        }
        for (i=0; i<N_V; i++)
            swap(&mask[i], &mask1[i]);
        pthread_barrier_wait(&bar);
    }
    
    pthread_exit(NULL);
}


int main(){
    int j, p;
    long i;
    
    for (p=0; p<N_V; p++)
    {
        for (j=0; j<N_V; j++)
        {
            m[p][j] = rand() % 40;
        }
    }
    
    clock_t begin = clock();
    pthread_t thread[NUM_THREADS];
    for (i=0; i<NUM_THREADS; i++)
    {
        pthread_create(&thread[i], NULL, ini, (void *) i);
    }
    for (i=0; i<NUM_THREADS; i++)
        pthread_join(thread[0], NULL);
    d[0] = 0;
    mask[0] = 1;
    flag = 1;
    
    for (i=0; i<NUM_THREADS; i++)
    {
        pthread_create(&thread[i], NULL, helper, (void *) i);
    }
    
    for (i=0; i<NUM_THREADS; i++)
        pthread_join(thread[i], NULL);
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    printf("%f\n", time);
    
    return 0;
    
    
    
    
}
 

/*
void *helper(void *arg){
    long taskid;
    taskid = (int) arg;
    
    long first, last;
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    long i, j, k;

    for (k=0; k<N_V; k++)
    {
        for (i=first; i<last; i++)
        {
         for (j=0; j<N_V; j++)
         {
            pthread_mutex_lock(&mutex);
            if (d[i] == INT_MAX || d[i] > d[j] + m[j][i])
            {
                d[i] = d[j] + m[j][i];
            }
            pthread_mutex_unlock(&mutex);
         }
        }
        pthread_barrier_wait(&bar);
    }
    return 0;
}


    
int main(){
    int j, p;
    long i;
    for (p=0; p<N_V; p++)
    {
        for (j=0; j<N_V; j++)
        {
            m[p][j] = rand() % 40;
        }
    }
    
    
    clock_t begin = clock();
    
    pthread_t thread[NUM_THREADS];
    
    for (i=0; i<N_V; i++)
    {
        d[i] = INT_MAX;
    }
    d[0] = 0;
    
    
    for (i=0; i<NUM_THREADS; i++)
    {
        pthread_create(&thread[i], NULL, helper, (void *) i);
    }
    
    for (i=0; i<NUM_THREADS; i++)
        pthread_join(thread[i], NULL);
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    printf("%f\n", time);
    
    pthread_exit(NULL);
}
*/
/*
int main(){
    int j, p, i;
    for (p=0; p<N_V; p++)
    {
        for (j=0; j<N_V; j++)
        {
            m[p][j] = rand() % 40;
        }
    }
    clock_t begin = clock();
    
    for (j=0; j<N_V; j++)
        d[j] = INT_MAX;
    d[0] = 0;
    
    for (p=0; p<N_V; p++)
    {
        for (i=0; i<N_V; i++)
        {
            for (j=0; j<N_V; j++)
            {
                if (d[i] == INT_MAX || d[i] > d[j] + m[j][i])
                {
                    d[i] = d[j] + m[j][j];
                }
            }
            
        }
    }
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    printf("%f\n", time);

    return 0;
}




*/
