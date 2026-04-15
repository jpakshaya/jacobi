#include <stdio.h>
#include <cuda.h>
#include <time.h>

#define N 1000
#define M 1000
#define error 0.001


__global__ void heat_transfer(float *old_grid, float *new_grid, float *d_diff){

    __shared__ float tile[18][18];
    __shared__ float block_diffs[256];

    int tx= threadIdx.x;
    int ty= threadIdx.y;
    int i= blockIdx.y*blockDim.y + threadIdx.y;
    int j= blockIdx.x*blockDim.x + threadIdx.x;

    if(i<N && j<M){
        tile[ty+1][tx+1]= old_grid[i*M+j];
    }
    else{
        tile[ty+1][tx+1]= 0.0;
    }

    if (tx == 0) {
        if (j > 0) {
            tile[ty + 1][0] = old_grid[i * M + (j - 1)];
        } else {
            tile[ty + 1][0] = 0.0f; // Boundary condition
        }
    }

    // 2. Right Border: Only the last column of threads in the block (tx == 15)
    if (tx == 15) {
        if (j < M - 1) {
            tile[ty + 1][17] = old_grid[i * M + (j + 1)];
        } else {
            tile[ty + 1][17] = 0.0f; // Boundary condition
        }
    }

    // 3. Top Border: Only the first row of threads in the block (ty == 0)
    if (ty == 0) {
        if (i > 0) {
            tile[0][tx + 1] = old_grid[(i - 1) * M + j];
        } else {
            tile[0][tx + 1] = 0.0f; // Boundary condition
        }
    }

    // 4. Bottom Border: Only the last row of threads in the block (ty == 15)
    if (ty == 15) {
        if (i < N - 1) {
            tile[17][tx + 1] = old_grid[(i + 1) * M + j];
        } else {
            tile[17][tx + 1] = 0.0f; // Boundary condition
        }
    }

    __syncthreads();

    float diff = 0;
    if (i > 0 && i < N - 1 && j > 0 && j < M - 1) {
        float res = (tile[ty][tx+1] + tile[ty+2][tx+1] + 
                     tile[ty+1][tx] + tile[ty+1][tx+2]) / 4.0f;
        new_grid[i * M + j] = res;
        diff = fabsf(res - tile[ty + 1][tx + 1]);
    }

    int tid = ty * 16 + tx;
    block_diffs[tid] = diff;
    __syncthreads();

    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            block_diffs[tid] = fmaxf(block_diffs[tid], block_diffs[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        // AtomicMax for floats trick using integer representation
        atomicMax((int*)d_diff, __float_as_int(block_diffs[0]));
    }
}

__global__ void swap_grids(float *old_grid, float *new_grid){

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N && j < M){
        old_grid[i*M+j] = new_grid[i*M+j];
    }
}   

void print_grid(float *grid){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < M; j++){
            printf("%f\t", grid[i*M+j]);
        }
        printf("\n");
    }
}

int main(){
    printf("Main\n");

    clock_t cpu_start= clock();

    float *old_grid = (float*)malloc(N * M * sizeof(float));
    float *new_grid = (float*)malloc(N * M * sizeof(float));

    for(int i=0;i<N;i++){
        for(int j=0;j<M;j++){
            if(i==0){
                old_grid[i*M+j]=100;
                new_grid[i*M+j]=100;
            }
            else{
                old_grid[i*M+j]=0;
                new_grid[i*M+j]=0;
            }
        }
    }
    printf("initialized\n");

    clock_t cpu_end= clock();
    double cpu_time= (double)(cpu_end - cpu_start)/ CLOCKS_PER_SEC;

    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);
    cudaEventRecord(gpu_start);

    //gpu initialization 
    float *d_old, *d_new, *d_diff;
    cudaMalloc(&d_old, N*M*sizeof(float));
    cudaMalloc(&d_new, N*M*sizeof(float));
    cudaMalloc(&d_diff, sizeof(float));

    cudaMemcpy(d_old,old_grid,N*M*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_new,new_grid,N*M*sizeof(float),cudaMemcpyHostToDevice);

    dim3 block(16,16);
    dim3 grid_dim((M+15)/16,(N+15)/16);

    float max_diff = 1.0;
    int iter = 0;

    while(max_diff > error){
        cudaMemset(d_diff, 0, sizeof(float));
        heat_transfer<<<grid_dim,block>>>(d_old,d_new,d_diff);
        cudaDeviceSynchronize();

        cudaMemcpy(&max_diff, d_diff, sizeof(float), cudaMemcpyDeviceToHost);
        float *temp= d_old;
        d_old= d_new;
        d_new= temp;
        iter++;
        //swap_grids<<<grid_dim, block>>>(d_old, d_new);
        cudaDeviceSynchronize();

    }

    cudaMemcpy(old_grid, d_old, N*M*sizeof(float), cudaMemcpyDeviceToHost);
    //printf("Old Grid: \n");
    //print_grid(old_grid);
    //printf("New Grid: \n");
    //print_grid(new_grid);

    cudaFree(d_old);
    cudaFree(d_new);
    cudaFree(d_diff);

    cudaEventRecord(gpu_stop);
    cudaEventSynchronize(gpu_stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop);
    double gpu_time= gpu_ms/1000;

    double total_time = cpu_time + gpu_time;

    printf("CPU Time:  %f seconds\n", cpu_time);
    printf("GPU Time:       %f seconds\n", gpu_time);
    printf("Total Time:     %f seconds\n", total_time);

    return 0;
}

