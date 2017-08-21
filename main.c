//
//  main.c
//  bf
//
//  Created by 安睿健 on 2017-08-20.
//  Copyright © 2017 安睿健. All rights reserved.
//


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


















/*
void *tfunc(void *arg)
{
    long id = (long ) arg;
    long first, last;
    
    first = (id * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    long q,p;
    for (p=first; p<last; p++)
        for (q=0; q<N_V; q++)
            if (m[N_V][p] == INT_MAX && m[N_V][p] > m[N_V][q] + m[N_V][q] + m[q][p])
                m[N_V][p] = m[N_V][q] + m[N_V][q] + m[q][p];
    free(arg);
    pthread_exit(NULL);
}
*/



/*

int main()
{
    pthread_t w[NUM_THREADS];
 
    int i, j;
    for (i=0; i<N_V; i++)
        m[N_V][i] = INT_MAX;
        for (j=0; j<N_V; j++)
            m[i][j] = rand() % 40;
    m[N_V][0] = 0;
    
    clock_t begin = clock();
    long n;
    long *taskid;
    for (n=0; n<NUM_THREADS; n++)
        taskid = (long *) malloc(sizeof(long));
        *taskid = n;
        pthread_create(&w[i], NULL, tfunc, (void *) taskid);
   
    clock_t end = clock();
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    printf("%f", time);
    
    pthread_exit(NULL);

}



*/















/*
int main()
{
    struct combo c[NUM_THREADS];
    pthread_t worker[NUM_THREADS];
    
    int i, j, l;
    for (i=0; i<NUM_THREADS; i++)
        for (j=0; j<N_V; j++)
            c[i].d[j] = INT_MAX;
            c[i].d[0] = 0;
            for (l=0; l<N_V/NUM_THREADS; l++)
                c[i].m[l][j] = rand() % 40;
    
    
    int k;
    for (k=0; k<NUM_THREADS; k++)
        pthread_create(&worker[k], NULL, new, c[k]);
    
    pthread_exit(NULL);
    return 0;
    
}
*/

/*
int main()

{
    struct combo c;
    
    int e, r, t;
    for (e=0; e<N_V; e++)
        c.d[e] = INT_MAX;
    c.d[0] = 0;
    
    for (r=0; r<N_V; r++)
        for (t=0; t<N_V; t++)
            c.m[r][t] = rand() % 20;


    int i, j, k;
    int f;
    
    
    for (f=0; f<N_V; f++)
        for (i=0; i<N_V; i++)
            for (j=0; j<N_V; j++)
                if (c.d[i] == INT_MAX && c.d[i] > c.d[j] + c.m[j][i])
                    c.d[i] = c.d[j] + c.m[j][i];
    for (k=0; k<N_V; k++)
        printf("%d\n", c.d[k]);
                
    return 0;
}


*/




