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
#define NUM_THREADS 48
#define N_V 1024
struct combo{
    int m[N_V][N_V];
    int d[N_V];
    int x[N_V];
    int x1[N_V];
    int id;
    int v;
    int y;
    int z;
};

struct sma{
    int m[N_V/NUM_THREADS][N_V];
    int id;
};

int flag = 1;
int mask[N_V];

int m[N_V][N_V];
int d[N_V];

int mask1[N_V];











/*


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
    for (k=0; k<N_V; k++){
        for (i=0; i<NUM_THREADS; i++)
            a = (struct sma *) malloc(sizeof(struct sma));
            for (p=0; p<N_V/NUM_THREADS; p++)
                for (j=0; j<N_V; j++)
                {
                    a->m[p][j] = rand() % 40;
                }
            a->id = i;
            pthread_create(&thread[i], NULL, helper, (void *) a);
        
        for (i=0; i<NUM_THREADS; i++)
            pthread_join(thread[i], NULL);
        
        
        if (flag == 0)
            break;
        flag = 0;
        
        for (i=0; i<N_V; i++)
            swap(&mask[i], &mask1[i]);
        for (i=0; i<N_V; i++)
            swap(&mask[i], &mask1[i]);
        
    }
    
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    printf("%f\n", time);
    
    return 0;
    
    
    
    
}
*/

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
        for (i=0; i<NUM_THREADS; i++)
            a = (struct sma *) malloc(sizeof(struct sma));
            for (p=0; p<N_V/NUM_THREADS; p++)
                for (j=0; j<N_V; j++)
                {
                    a->m[p][j] = rand() % 40;
                }
            a->id = i;
            pthread_create(&thread[i], NULL, helper, (void *) a);
        
        for (i=0; i<NUM_THREADS; i++)
            pthread_join(thread[i], NULL);
    
    }
    
    
    clock_t end = clock();
    
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
    
    
    printf("%f\n", time);
    
    return 0;
    
    
    
    
}


/*

void *helper0(void *c){
    int taskid;
    struct combo *my_data;
    my_data = (struct combo *) c;
    taskid = my_data->id;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++){
        for (j=0; j<N_V; j++){
            if (my_data->d[i] == INT_MAX && my_data->d[i] > my_data->d[j] + my_data->m[j][i])
                my_data->d[i] = my_data->d[j] + my_data->m[j][i];
        }
    }
    pthread_exit(NULL);
}

void *helper1(void *c){
    int taskid;
    struct combo *my_data;
    my_data = (struct combo *) c;
    taskid = my_data->v;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++){
        for (j=0; j<N_V; j++){
            if (my_data->d[i] == INT_MAX && my_data->d[i] > my_data->d[j] + my_data->m[j][i])
                my_data->d[i] = my_data->d[j] + my_data->m[j][i];
        }
    }

    pthread_exit(NULL);
}

void *helper2(void *c){
    int taskid;
    struct combo *my_data;
    my_data = (struct combo *) c;
    taskid = my_data->y;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++){
        for (j=0; j<N_V; j++){
            if (my_data->d[i] == INT_MAX && my_data->d[i] > my_data->d[j] + my_data->m[j][i])
                my_data->d[i] = my_data->d[j] + my_data->m[j][i];
        }
    }
    pthread_exit(NULL);
}

*/


/*

void *helper3(void *c){
    int taskid;
    struct combo *my_data;
    my_data = (struct combo *) c;
    taskid = my_data->z;
    
    int first, last;
    
    first = (taskid * N_V)/NUM_THREADS;
    last = first + N_V/NUM_THREADS;
    
    int i, j;
    
    for (i=first; i<last; i++){
        for (j=0; j<N_V; j++){
            if (my_data->d[i] == INT_MAX && my_data->d[i] > my_data->d[j] + my_data->m[j][i])
                my_data->d[i] = my_data->d[j] + my_data->m[j][i];
        }
    }

    pthread_exit(NULL);
}




int main(){
    pthread_t threads[NUM_THREADS];
    struct combo *tdata = NULL;
 
    tdata = (struct combo *) malloc(sizeof(struct combo));
    int e, r, m;
    for (e=0; e<N_V; e++)
        tdata->d[e] = INT_MAX;
    tdata->d[0] = 0;
    
    for (r=0; r<N_V; r++){
        for (m=0; m<N_V; m++){
            tdata->m[r][m] = rand() % 20;
        }
    }
    tdata->id = 0;
    tdata->v = 1;
    tdata->y = 2;
    tdata->z = 3;
    
    clock_t begin = clock();

    for (r=0; r<N_V; r++){
        pthread_create(&threads[0], NULL, helper0, (void *) tdata);
        pthread_create(&threads[1], NULL, helper1, (void *) tdata);
        pthread_create(&threads[2], NULL, helper2, (void *) tdata);
        pthread_create(&threads[3], NULL, helper3, (void *) tdata);
        
        pthread_join(threads[0],NULL);
        pthread_join(threads[1], NULL);
        pthread_join(threads[2],NULL);
        pthread_join(threads[3], NULL);
    }
    clock_t end = clock();
    double time = (double)(end - begin)/CLOCKS_PER_SEC;
 

    printf("%f\n", time);
    
    return 0;
}
*/
/*
int main()

{   clock_t begin = clock();
    int a,b,c;
    struct combo *co = NULL;
    co = (struct combo *) malloc(sizeof(struct combo));
    for (a=0; a<N_V; a++)
        co->d[a] = INT_MAX;
    co->d[0] = 0;
    
    for (b=0; b<N_V; b++){
        for (c=0; c<N_V; c++){
            co->m[b][c] = rand() % 40;
        }
    }
    
    int i, j;
    int f;
    
    for (f=0; f<N_V; f++)
        for (i=0; i<N_V; i++){
            for (j=0; j<N_V; j++){
                if (co->d[i] == INT_MAX && co->d[i] > co->d[j] + co->m[j][i]){
                    co->d[i] = co->d[j] + co->m[j][i];
                }
            }
        }
    clock_t end = clock();
    double spent = (double)(end- begin)/CLOCKS_PER_SEC;
    
    printf("%f\n", spent);
    return 0;
}

*/

/*
int main()
{
    struct combo *a = NULL;
    int p, j, k, i;
    a = (struct combo *) malloc(sizeof(struct combo));
    
    
    clock_t begin = clock();
    
    for (p=0; p<N_V/NUM_THREADS; p++)
        for (j=0; j<N_V; j++)
        {
            a->m[p][j] = rand() % 40;
        }
    for (k=0; k<N_V; k++)
        d[k] = INT_MAX;
    d[0] = 0;
    for (p=0; p<N_V; p++){
        for (i=0; i<N_V;i++)
            for (k=0; k<N_V; k++)
            {
                if (d[i] ==INT_MAX && d[i]> d[k] + a->m[k][i])
                    d[i] = d[k] + a->m[k][i];
            }
    }
    clock_t end = clock();
    double spent = (double)(end- begin)/CLOCKS_PER_SEC;
    
    printf("%f\n", spent);
    return 0;

}
*/
