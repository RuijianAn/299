

#include <pthread.h>
#include <stdio.h>
#include <limits.h>
#include <stdlib.h>
#include <time.h>
#define NUM_THREADS	4
#define N_V 32

struct combo{
    int m[N_V][N_V];
    int d[N_V];
    int id;
};


void *helper(void *c){
    int taskid;
    struct combo *my_data;
    my_data = (struct combo *) c;
    taskid = my_data->id;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++)
        for (j=0; j<N_V; j++)
            if (my_data->d[i] == INT_MAX && my_data->d[i] > my_data->d[j] + my_data->m[j][i])
                my_data->d[i] = my_data->d[j] + my_data->m[j][i];
    free(c);
    pthread_exit(NULL);
}
int main(){
    
    int t,rc;
    
    /* Record the starting time */
    clock_t begin = clock();
    pthread_t threads[NUM_THREADS];
    struct combo *tdata = NULL;
    
    
    /* Initialize the matrix and dist */
    for (t=0;t<NUM_THREADS;t++)
        tdata = (struct combo *) malloc(sizeof(struct combo));
    tdata->id = t;
    int e, r, m;
    for (e=0; e<N_V; e++)
        tdata->d[e] = INT_MAX;
    tdata->d[0] = 0;
    
    
    for (r=0; r<N_V; r++)
        for (m=0; m<N_V; m++)
            tdata->m[r][m] = rand() % 20;
    
    /* Create multiple threads */
    rc = pthread_create(&threads[t], NULL, helper, (void *) tdata);
    
    if (rc){
        
        exit(-1);
    }
    
    clock_t end = clock();
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    /* Print the total running time */
    printf("%f\n", time);

    return 0;
}





