#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 1000
#define M 1000
#define error 0.001

float old_grid[N][M];
float new_grid[N][M];
float difference;
float max_diff=1.0;

void heat_spread(){

    while(max_diff>error){

        max_diff=0.0;
        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                if(i>0 && i<N-1 && j>0 && j<M-1){
                    new_grid[i][j]= (old_grid[i-1][j]+ old_grid[i+1][j]+ old_grid[i][j-1]+ old_grid[i][j+1])/4;
                    difference= fabsf(new_grid[i][j]-old_grid[i][j]);

                    if(difference> max_diff){
                        max_diff= difference;
                    }
                }
            }
        }

        for(int i=0;i<N;i++){
            for(int j=0;j<M;j++){
                old_grid[i][j]= new_grid[i][j];
            }
        }
    }
    
}

float print_grid(float grid[N][M]){
    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            printf("%f\t",grid[i][j]);
        }
        printf("\n");
    }
}

int main(){

    clock_t start= clock();

    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            if(i==0){
                old_grid[i][j]=100;
                new_grid[i][j]=100;
            }
            else{
                old_grid[i][j]=0;
                new_grid[i][j]=0;
            }
        }
    }

    heat_spread();
    // printf("Old Grid: \n");
    // print_grid(old_grid);
    // printf("New Grid: \n");
    // print_grid(new_grid);
    
    clock_t end= clock();

    double time_taken= (double)(end- start)/CLOCKS_PER_SEC;
    printf("\nTime taken by serial code: %f seconds\n",time_taken);

    return 0;

    printf("Grid size improved");
    for(int i=0;i<N;i++){
        printf(" "); 
        printf("Assigned value: ");
        for(int j=0;j<N;j++){
            printf("Assigned value to j= ");
            break;
            //break the loop
        }
    }
}








