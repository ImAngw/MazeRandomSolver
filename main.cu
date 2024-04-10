#include "cstdio"
#include "maze_func.cuh"


__constant__ int x_max = XMAX;

__global__ void rand_solver_gpu(const short *d_lin_maze, int x_start, int y_start, int x_ext, int y_ext, int *flag, int n){

    /*
     * It generates N particles (starting from (x_start, y_start)) and for all of them the functions generates
     * a random path until one of them reaches the exit of the maze (x_ext, y_ext).
     */

    unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

    if(idx < n){
        float rand_choice;
        int d_x = x_start, d_y = y_start, temp;
        int n_steps = 0;

        curandState state;
        curand_init(clock64(), idx, 0, &state);

        while(*flag != 1){  // You can choose between *flag != 1 AND n_steps < 1000
            n_steps++;
            rand_choice = curand_uniform(&state) * (float) (d_lin_maze[11 * d_y * x_max + 11 * d_x + 2]) + 1 ;

            temp = (int)d_lin_maze[11 * d_y * x_max + 11 * d_x + 2 + 2 * (int) rand_choice - 1];
            d_y = (int)d_lin_maze[11 * d_y * x_max + 11 * d_x + 2 + 2 * (int) rand_choice];
            d_x = temp;

            if(d_x == x_ext && d_y == y_ext) {
                atomicExch(flag, 1);
                //printf(" \nGPU solution found (B %d T %d) in:\n - %d steps\n",blockIdx.x, idx, n_steps);
            }
            __syncthreads();
        }
    }
}


int main() {


    // Start and exit tiles
    Point start, *solution;
    start.x = 0; start.y = 3;

    // Maze matrix
    char **maze;

    // Time variables
    float milliseconds = 0;
    cudaEvent_t t_start, t_stop;

    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);


    // Host and device flags
    int *h_flag, *d_flag;
    h_flag = (int*) malloc(sizeof(int));
    *h_flag = 0;

    // Linearized maze
    short h_lin_maze[XMAX * YMAX * 11], *d_lin_maze;

    int k = 0, n_ways;
    Adjacents *adjac;
    adjac = (Adjacents*) malloc(sizeof(Adjacents));

    // Starting tile of the maze
    solution = (Point*) malloc(sizeof (Point));
    if (solution == nullptr) return 1;

    // Maze initialization
    maze = maze_init(start, solution, XMAX, YMAX);
    //print_maze(maze, XMAX, YMAX);



    printf("\n Maze dim: %dx%d\n", XMAX, YMAX);
    printf(" Number of particles: %d\n", N);
    printf("---------------------------------\n");


    // Here I linearize the maze
    for (int i = 0; i < YMAX; i++){
        for (int j = 0; j < XMAX; j++){
            h_lin_maze[k] = (short)j;
            h_lin_maze[k+1] = (short)i;
            k += 2;
            if (maze[i][j] == WAY || maze[i][j] == START){
                n_ways = find_ways(maze, XMAX, YMAX, adjac, j, i);
                h_lin_maze[k] = (short)n_ways;
                k++;
                for (int t = 0; t < 4; t++){
                    if (t < n_ways){
                        h_lin_maze[k] = (short)adjac->x[t];
                        h_lin_maze[k+1] = (short)adjac->y[t];
                    }
                    else{
                        h_lin_maze[k] = (short)0;
                        h_lin_maze[k+1] = (short)0;
                    }

                    k += 2;
                }
            }
            else{
                for (int t = 0; t < 9; t++){
                    h_lin_maze[k] = (short)0;
                    k++;
                }
            }
        }
    }

    cudaMalloc((void**) &d_flag, sizeof(int));
    cudaMalloc((void**)&d_lin_maze, XMAX * YMAX * 11 * sizeof(short));
    cudaMemcpy(d_lin_maze, h_lin_maze, XMAX * YMAX * 11 * sizeof(short), cudaMemcpyHostToDevice);



    // You can execute these lines if you want to collect times of the sequential version of the Maze RANDOM solver
    /*
    int blockSizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_particles[] = {N/20, N/10, N/5, 2*N/5, N/2, 3*N/4, N};
    float times[7][8];
    float cpu_times[7];

    for (int i = 0; i < 7; i++){

        milliseconds = 0.0;
        cudaEventRecord(t_start);

        rand_solver_cpu(h_lin_maze, solution->x, solution->y, start.x, start.y, n_particles[i]);
        cudaEventRecord(t_stop);

        cudaEventSynchronize(t_stop);
        cudaEventElapsedTime(&milliseconds, t_start, t_stop);
        cpu_times[i] = milliseconds;
    }

    // Print times on screen
    for (int i = 0; i < 7; i++){
        printf("%f, ",cpu_times[i]);
    }
    printf("\n");
    */

    // You can execute these lines if you want to collect times of the parallel version of the Maze RANDOM solver
    /*
    // Arrays for data collection
    int blockSizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int n_particles[] = {N/20, N/10, N/5, 2*N/5, N/2, 3*N/4, N};
    float times[7][8];

    for (int i = 0; i < 8; ++i){
        for (int j = 0; j < 7; ++j){
            milliseconds = 0.0;
            cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

            cudaEventRecord(t_start);
            rand_solver_gpu<<<(int)(n_particles[j] + blockSizes[i] -1)/blockSizes[i], blockSizes[i]>>>(d_lin_maze,
                                                                              solution->x,
                                                                              solution->y,start.x,
                                                                              start.y, d_flag,
                                                                              n_particles[j]);

            cudaDeviceSynchronize();
            cudaEventRecord(t_stop);

            cudaEventSynchronize(t_stop);
            cudaEventElapsedTime(&milliseconds, t_start, t_stop);
            times[j][i] = milliseconds;
        }
    }

    // Print times on screen

    printf("\n");
    for (int i = 0; i < 8; ++i){
        //printf("NTHR %d ", blockSizes[i]);
        for (int j = 0; j < 7; ++j){
            printf("%f, ",times[j][i]);
        }
        printf("\n");
    }
    */





    // Free all dynamic variables (both on CPU and GPU)
    for (int i = 0; i < YMAX; ++i){
        free(maze[i]);
    }

    free(maze);
    free(solution);
    free(h_flag);
    free(adjac);


    // CUDA FREE MEMORY
    cudaFree(d_lin_maze);
    cudaFree(d_flag);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);

    return 0;
}
