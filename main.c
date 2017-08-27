

#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#define NUM_THREADS 4
#define N_V 1024


struct sma{
    int m[N_V/NUM_THREADS][N_V];
    int id;
};

int flag = 1;
int mask[N_V];
int mask1[N_V];


int d[N_V];









/* Work-efficient multithread CPU version */

void *helper(void *c){
    int taskid;
    
    struct sma *my_data;
    my_data = (struct sma *) c;
    taskid = my_data->id;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++)
        for (j=0; j<N_V; j++){
            if (mask[j] == 1){
                if (d[i] == INT_MAX && d[i] > d[j] + my_data->m[j][i])
                    d[i] = d[j] + my_data->m[j][i];
                    mask1[i] = 1;
                    flag = 1;
            }
        }
    free(c);
    pthread_exit(NULL);
}

void swap(int *a, int *b)
{
    int temp;
    
    temp = *b;
    *b   = *a;
    *a   = temp;
}


int main(){
    struct sma *a = NULL;
    int i, j, k, p;
    
    for (i=0; i<N_V; i++)
    {
        d[i] = INT_MAX;
        mask[i] = 0;
    }
    
    d[0] = 0;
    mask[0] = 1;
    
    pthread_t thread[NUM_THREADS];
    
    clock_t begin = clock();
    flag = 1;
    
    /* Outer loop */
    for (k=0; k<N_V; k++){
        /* Create multiple threads */
        for (i=0; i<NUM_THREADS; i++)
            a = (struct sma *) malloc(sizeof(struct sma));
            for (p=0; p<N_V/NUM_THREADS; p++)
                for (j=0; j<N_V; j++)
                {
                    a->m[p][j] = rand() % 40;
                }
            a->id = i;
            pthread_create(&thread[i], NULL, helper, (void *) a);
        
        /* Synchronization */
        for (i=0; i<NUM_THREADS; i++)
            pthread_join(thread[i], NULL);
        
        
        if (flag == 0)
            break;
        flag = 0;
        
        for (i=0; i<N_V; i++)
            swap(&mask[i], &mask1[i]);
    }
    
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    printf("%f\n", time);
    
    return 0;
   
    
    
}

/* Work inefficent version */
void *helper(void *c){
    int taskid;
    struct sma *my_data;
    my_data = (struct sma *) c;
    taskid = my_data->id;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++)
        for (j=0; j<N_V; j++)
            if (d[i] == INT_MAX && d[i] > d[j] + my_data->m[j][i])
                d[i] = d[j] + my_data->m[j][i];
    free(c);
    pthread_exit(NULL);
}


int main(){
    struct sma *a = NULL;
    int i, j, k, p;
    
    for (i=0; i<N_V; i++)
    {
        d[i] = INT_MAX;
    }
    
    d[0] = 0;
    pthread_t thread[NUM_THREADS];
        clock_t begin = clock();
    
    for (k=0; k<N_V; k++){
        for (i=0; i<NUM_THREADS; i++){
            a = (struct sma *) malloc(sizeof(struct sma));
            for (p=0; p<N_V/NUM_THREADS; p++)
                for (j=0; j<N_V; j++)
                {
                    a->m[p][j] = rand() % 40;
                }
            a->id = i;
            pthread_create(&thread[i], NULL, helper, (void *) a);
        }
        for (i=0; i<NUM_THREADS; i++)
            pthread_join(thread[i], NULL);
    
    }
    
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    printf("%f\n", time);
    
    return 0;
    
    
    
    
}


