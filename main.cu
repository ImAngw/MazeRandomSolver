#include "cstdio"
#include "maze_func.cuh"
#include "ctime"


#define XDIM 500
#define YDIM 500
#define NPART 1000
#define MAX_SIZE_PATH 100
#define BLOCKDIM 256
#define CONTROL_NUMBER 5
#define STEPS_ITER 50


__constant__ float steps_per_iteration = 50;

__global__ void RandSolver(const short *d_lin_maze, int x_start, int y_start, int x_ext, int y_ext, int *flag, int n,
                           int x_max){

    /*
     * It generates N particles (starting from (x_start, y_start)) and for all of them the functions generates
     * a random path until one of them reaches the exit of the maze (x_ext, y_ext).
     */

    unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

    if(idx < n){
        float rand_choice;
        int d_x = x_start, d_y = y_start, temp;
        int n_steps = 0;
        int current_y_pos;

        curandState state;
        curand_init(clock64(), idx, 0, &state);

        while(*flag != 1){  // You can choose between *flag != 1 OR n_steps < 1000

            current_y_pos = 11 * d_y * x_max + 11 * d_x + 2;
            rand_choice = curand_uniform(&state) * (float) (d_lin_maze[current_y_pos]) + 1;

            temp = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice - 1];
            d_y = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice];
            d_x = temp;

            if(d_x == x_ext && d_y == y_ext) {
                atomicExch(flag, 1);
                //printf(" \nGPU solution found (B %d T %d) in:\n - %d steps\n", blockIdx.x, idx, n_steps);
            }

            n_steps++;
            __syncthreads();
        }
    }
}

__global__ void OneStepMaze(const short *d_lin_maze, int x_ext, int y_ext, int *flag, int n,
                            int *x_array, int *y_array, int x_max){

    /*
     * It generates N particles (starting from (x_start, y_start)) and for all of them the functions generates
     * a random path until one of them reaches the exit of the maze (x_ext, y_ext).
     */

    unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

    if(idx < n){
        float rand_choice;
        int n_steps = 0;
        int temp_x = x_array[idx], temp_y = y_array[idx];
        int current_y_pos;

        curandState state;
        curand_init(clock64(), idx, 0, &state);

        while (n_steps <  2 * x_max /*&& *flag != 1*/){
            current_y_pos = 11 * temp_y * x_max + 11 * temp_x + 2;
            rand_choice = curand_uniform(&state) * (float) (d_lin_maze[current_y_pos]) + 1;

            temp_x = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice - 1];
            temp_y = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice];

            if(temp_x == x_ext && temp_y == y_ext) atomicExch(flag, 1);

            n_steps++;
            //__syncthreads();
        }

        x_array[idx] = temp_x;
        y_array[idx] = temp_y;
    }
}


__global__ void OneStepMazeWithPath(const short *d_lin_maze, int x_ext, int y_ext, int *flag, int n,
                            int *x_array, int *y_array, int x_max, int *paths_array){

    /*
     * It generates N particles (starting from (x_start, y_start)) and for all of them the functions generates
     * a random path until one of them reaches the exit of the maze (x_ext, y_ext).
     */

    unsigned int idx = blockIdx.x *blockDim.x + threadIdx.x;

    if(idx < n){
        float rand_choice;
        int done_steps = (int) paths_array[idx * (MAX_SIZE_PATH + 2)];
        int n_steps = 0;
        int temp_x = x_array[idx], temp_y = y_array[idx];
        int current_y_pos;
        int local_path[2 * STEPS_ITER];

        //printf("%d %d\n", temp_x, temp_y);
        curandState state;
        curand_init(clock64(), idx, 0, &state);

        while (n_steps <  2 * x_max && *flag != 1){
            current_y_pos = 11 * temp_y * x_max + 11 * temp_x + 2;
            rand_choice = curand_uniform(&state) * (float) (d_lin_maze[current_y_pos]) + 1;

            if (n_steps + done_steps < MAX_SIZE_PATH){
                paths_array[(MAX_SIZE_PATH + 2) * idx + done_steps + 2 + n_steps] = (int)local_path[n_steps];
            }

            temp_x = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice - 1];
            temp_y = (int)d_lin_maze[current_y_pos + 2 * (int) rand_choice];

            //printf("%d) %d %d\n",n_steps, temp_x, temp_y);

            if(temp_x == x_ext && temp_y == y_ext){
                atomicExch(flag, 1);
                //printf("SOLUTION FOUND\n");
                paths_array[idx * (MAX_SIZE_PATH + 2) + 1] = (int) 1;
            }
            n_steps++;
            //__syncthreads();
        }

/*
        for (int i = 0; i < n_steps; i++){
            if (done_steps + i < MAX_SIZE_PATH){
                paths_array[(MAX_SIZE_PATH + 2) * idx + done_steps + 2 + i] = (int)local_path[i];
            }
            else break;
        }
*/
        done_steps += n_steps;
        paths_array[(MAX_SIZE_PATH + 2) * idx] = (int) done_steps;

        x_array[idx] = temp_x;
        y_array[idx] = temp_y;
    }
}










short *return_lin_maze(char **maze, int x_dim, int y_dim){

    int lin_maze_dim = x_dim * y_dim * 11;
    int k = 0, n_ways;
    short *lin_maze;
    Adjacents *adjac;

    adjac = (Adjacents*) malloc(sizeof(Adjacents));
    if (adjac == nullptr) {
        printf("Errore di allocazione della memoria\n");
        return nullptr;
    }

    lin_maze = (short *) malloc(lin_maze_dim * sizeof (short));
    if (lin_maze == nullptr) {
        printf("Errore di allocazione della memoria\n");
        return nullptr;
    }


    for (int i = 0; i < y_dim; i++){
        for (int j = 0; j < x_dim; j++){
            lin_maze[k] = (short)j;
            lin_maze[k+1] = (short)i;
            k += 2;
            if (maze[i][j] == WAY || maze[i][j] == START){
                n_ways = find_ways(maze, x_dim, y_dim, adjac, j, i);
                lin_maze[k] = (short)n_ways;
                k++;
                for (int t = 0; t < 4; t++){
                    if (t < n_ways){
                        lin_maze[k] = (short)adjac->x[t];
                        lin_maze[k+1] = (short)adjac->y[t];
                    }
                    else{
                        lin_maze[k] = (short)0;
                        lin_maze[k+1] = (short)0;
                    }

                    k += 2;
                }
            }
            else{
                for (int t = 0; t < 9; t++){
                    lin_maze[k] = (short)0;
                    k++;
                }
            }
        }
    }


    free(adjac);
    return lin_maze;
}

float **collect_speedup(int x_dim, int y_dim, int *n_part, int dim_n_part, int const *block_dim_array, int dim_block) {

    float **times;
    times = (float**) malloc(dim_n_part * sizeof(float*));
    for (int i = 0; i < dim_n_part; i++) times[i] = (float*) malloc(dim_block * sizeof(float));

    float time_seq;

    Point start;
    start.x = 0; start.y = 3;

    Point *solution;
    solution = (Point*) malloc(sizeof (Point));

    // Time variables
    float milliseconds;



    // Maze matrix
    char **maze = performance_maze_init(start, solution, x_dim, y_dim);

    // Linear maze
    short *h_lin_maze = return_lin_maze(maze, x_dim, y_dim);
    short *d_lin_maze;
    cudaMalloc((void**) &d_lin_maze, x_dim * y_dim * 11 * sizeof(short));
    cudaMemcpy(d_lin_maze, h_lin_maze, x_dim * y_dim * 11 * sizeof(short), cudaMemcpyHostToDevice);


    // Host and device flags
    int *h_flag, *d_flag;
    h_flag = (int*) malloc(sizeof(int));
    *h_flag = 0;
    cudaMalloc((void**) &d_flag, sizeof(int));
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


    // Device arrays of coordinates
    int *d_x_array, *d_y_array;

    for (int i = 0; i < dim_n_part; i++){

        cudaEvent_t t_start, t_stop;
        cudaEventCreate(&t_start);
        cudaEventCreate(&t_stop);

        // Host arrays of coordinates
        int *h_x_array = (int*) malloc(n_part[i] * sizeof(int));
        int *h_y_array = (int*) malloc(n_part[i] * sizeof(int));

        // Host Arrays initialization
        for (int j = 0; j < n_part[i]; j++){
            h_x_array[j] = solution->x;
            h_y_array[j] = solution->y;
        }


        cudaMalloc((void**) &d_x_array, n_part[i] * sizeof(int));
        cudaMalloc((void**) &d_y_array, n_part[i] * sizeof(int));

        cudaMemcpy(d_x_array, h_x_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_array, h_y_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);


        // SEQ TIME
        cudaEventRecord(t_start);
        rand_solver_cpu(h_lin_maze, x_dim, y_dim, start.x, start.y, n_part[i], h_x_array, h_y_array, 2 * x_dim);
        cudaEventRecord(t_stop);
        cudaEventSynchronize(t_stop);
        cudaEventElapsedTime(&time_seq, t_start, t_stop);
        //printf("SEQ %f\n", time_seq);

        for (int j = 0; j < n_part[i]; j++){
            h_x_array[j] = solution->x;
            h_y_array[j] = solution->y;
        }


        // Warm-up
        RandSolver<<<(5000 + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(d_lin_maze,
                                                                   solution->x,
                                                                   solution->y,start.x,
                                                                   start.y,
                                                                   d_flag,
                                                                   5000,
                                                                   x_dim);

        cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


        for (int j = 0; j < dim_block; j++){
            cudaEvent_t t_start1, t_stop1;
            cudaEventCreate(&t_start1);
            cudaEventCreate(&t_stop1);


            cudaEventRecord(t_start1);
            OneStepMaze<<< (n_part[i] + block_dim_array[j] - 1) / block_dim_array[j], block_dim_array[j] >>>(d_lin_maze,
                                                                                                             start.x,
                                                                                                             start.y,
                                                                                                             d_flag,
                                                                                                             n_part[i],
                                                                                                             d_x_array,
                                                                                                             d_y_array,
                                                                                                             x_dim);

            cudaEventRecord(t_stop1);
            cudaEventSynchronize(t_stop1);
            milliseconds = 0.0;
            cudaEventElapsedTime(&milliseconds, t_start1, t_stop1);
            //printf("MILL: %f\n", milliseconds);
            //printf("-----------\n");

            times[i][j] = time_seq / milliseconds;

            cudaEventDestroy(t_start1);
            cudaEventDestroy(t_stop1);

            cudaMemcpy(d_x_array, h_x_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_y_array, h_y_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);

        }

        free(h_x_array);
        free(h_y_array);

        cudaFree(d_x_array);
        cudaFree(d_y_array);

        cudaEventDestroy(t_start);
        cudaEventDestroy(t_stop);

    }

    //  FREE MEMORY

    for (int i = 0; i < y_dim; ++i){
        free(maze[i]);
    }

    free(maze);
    free(solution);
    free(h_flag);
    free(h_lin_maze);


    // CUDA FREE MEMORY
    cudaFree(d_lin_maze);
    cudaFree(d_flag);
    cudaFree(d_x_array);
    cudaFree(d_y_array);



    return times;
}

float **collect_times(int *n_part, int dim_n_part, int const *sizes_array, int dim_sizes, int max_iter, int block_dim = 64) {

    float **times;
    times = (float**) malloc(dim_sizes * sizeof(float*));
    int n_steps = 0;

    for (int i = 0; i < dim_sizes; i++){
        times[i] = (float*) malloc(dim_n_part * sizeof(float));
        for (int j = 0; j < dim_n_part; j++) times[i][j] = 0.0;
    }

    Point start;
    start.x = 0; start.y = 3;



    // Host and device flags
    int *h_flag, *d_flag;
    h_flag = (int*) malloc(sizeof(int));
    *h_flag = 0;
    cudaMalloc((void**) &d_flag, sizeof(int));
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);



    // Time variables
    float milliseconds;

    for (int k = 0; k < dim_sizes; k++){

        Point *solution;
        solution = (Point*) malloc(sizeof (Point));

        // Maze matrix
        char **maze = performance_maze_init(start, solution, sizes_array[k], sizes_array[k]);
        // Linear maze
        short *h_lin_maze = return_lin_maze(maze, sizes_array[k], sizes_array[k]);
        short *d_lin_maze;

        cudaMalloc((void**) &d_lin_maze, sizes_array[k] * sizes_array[k] * 11 * sizeof(short));
        cudaMemcpy(d_lin_maze, h_lin_maze, sizes_array[k] * sizes_array[k] * 11 * sizeof(short), cudaMemcpyHostToDevice);
        //printf("INIT\n");

        for (int i = 0; i < dim_n_part; i++){

            // Host arrays of coordinates
            int *h_x_array = (int*) malloc(n_part[i] * sizeof(int));
            int *h_y_array = (int*) malloc(n_part[i] * sizeof(int));

            // Device arrays of coordinates
            int *d_x_array, *d_y_array;

            // Host Arrays initialization
            for (int j = 0; j < n_part[i]; j++){
                h_x_array[j] = solution->x;
                h_y_array[j] = solution->y;
            }

            cudaMalloc((void**) &d_x_array, n_part[i] * sizeof(int));
            cudaMalloc((void**) &d_y_array, n_part[i] * sizeof(int));



            // Warm-up
            RandSolver<<<(5000 + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(d_lin_maze,
                                                                       solution->x,
                                                                       solution->y,start.x,
                                                                       start.y,
                                                                       d_flag,
                                                                       5000,
                                                                       sizes_array[k]);

            cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


            for (int w = 0; w < max_iter; w++){
                cudaMemcpy(d_x_array, h_x_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y_array, h_y_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);

                cudaEvent_t t_start1, t_stop1;
                cudaEventCreate(&t_start1);
                cudaEventCreate(&t_stop1);

                cudaEventRecord(t_start1);

                RandSolver<<< (n_part[i] + block_dim - 1) / block_dim, block_dim >>>(d_lin_maze,
                                                                                     solution->x,
                                                                                     solution->y,
                                                                                     start.x,
                                                                                     start.y,
                                                                                     d_flag,
                                                                                     n_part[i],
                                                                                     sizes_array[k]
                                                                                     );


                /*
                while (*h_flag != 1){
                    n_steps++;
                    OneStepMaze<<< (n_part[i] + block_dim - 1) / block_dim, block_dim >>>(d_lin_maze,
                                                                                          start.x,
                                                                                          start.y,
                                                                                          d_flag,
                                                                                          n_part[i],
                                                                                          d_x_array,
                                                                                          d_y_array,
                                                                                          sizes_array[k]);
                    cudaDeviceSynchronize();
                    cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

                }
                */


                cudaEventRecord(t_stop1);
                cudaEventSynchronize(t_stop1);
                milliseconds = 0.0;
                cudaEventElapsedTime(&milliseconds, t_start1, t_stop1);
                times[k][i] += milliseconds;

                *h_flag = 0;
                cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

                //printf("%f\n", milliseconds);

                cudaEventDestroy(t_start1);
                cudaEventDestroy(t_stop1);
            }

            times[k][i] = times[k][i] / (float) max_iter;
            printf("%f, ", times[k][i]);

            n_steps = 0;
            free(h_x_array);
            free(h_y_array);

            cudaFree(d_x_array);
            cudaFree(d_y_array);
        }
        printf("\n");

        for (int i = 0; i < sizes_array[k]; ++i){
            free(maze[i]);
        }

        free(maze);
        free(h_lin_maze);
        free(solution);

        cudaFree(d_lin_maze);
        //printf("* - ");
    }
    //printf("|\n");

    free(h_flag);
    cudaFree(d_flag);

    return times;
}

float **collect_times_with_path(int *n_part, int dim_n_part, int const *sizes_array, int dim_sizes, int max_iter, int block_dim = 64) {

    float **times;
    times = (float**) malloc(dim_sizes * sizeof(float*));
    int n_steps = 0;

    for (int i = 0; i < dim_sizes; i++){
        times[i] = (float*) malloc(dim_n_part * sizeof(float));
        for (int j = 0; j < dim_n_part; j++) times[i][j] = 0.0;
    }

    Point start;
    start.x = 0; start.y = 3;



    // Host and device flags
    int *h_flag, *d_flag;
    h_flag = (int*) malloc(sizeof(int));
    *h_flag = 0;
    cudaMalloc((void**) &d_flag, sizeof(int));
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);



    // Time variables
    float milliseconds;

    for (int k = 0; k < dim_sizes; k++){

        Point *solution;
        solution = (Point*) malloc(sizeof (Point));

        // Maze matrix
        char **maze = performance_maze_init(start, solution, sizes_array[k], sizes_array[k]);
        // Linear maze
        short *h_lin_maze = return_lin_maze(maze, sizes_array[k], sizes_array[k]);
        short *d_lin_maze;

        cudaMalloc((void**) &d_lin_maze, sizes_array[k] * sizes_array[k] * 11 * sizeof(short));
        cudaMemcpy(d_lin_maze, h_lin_maze, sizes_array[k] * sizes_array[k] * 11 * sizeof(short), cudaMemcpyHostToDevice);
        //printf("INIT\n");

        for (int i = 0; i < dim_n_part; i++){

            // Host arrays of coordinates
            int *h_x_array = (int*) malloc(n_part[i] * sizeof(int));
            int *h_y_array = (int*) malloc(n_part[i] * sizeof(int));

            // Device arrays of coordinates
            int *d_x_array, *d_y_array;

            // Host Arrays initialization
            for (int j = 0; j < n_part[i]; j++){
                h_x_array[j] = solution->x;
                h_y_array[j] = solution->y;
            }

            cudaMalloc((void**) &d_x_array, n_part[i] * sizeof(int));
            cudaMalloc((void**) &d_y_array, n_part[i] * sizeof(int));

            int *h_paths_array;
            h_paths_array = (int*) malloc((MAX_SIZE_PATH + 2) * n_part[i] * sizeof(int));

            for (int ii = 0; ii < (MAX_SIZE_PATH + 2) * n_part[i]; ii++) {
                if (ii % (MAX_SIZE_PATH + 2) == 0){
                    h_paths_array[ii] = (int) 0;
                    h_paths_array[ii + 1] = (int) 0;
                    ii += 1;
                }

                else h_paths_array[ii] = (int) CONTROL_NUMBER;
            }

            int *d_paths_array;
            cudaMalloc((void**) &d_paths_array, (MAX_SIZE_PATH + 2) * n_part[i] * sizeof(int));



            // Warm-up
            RandSolver<<<(5000 + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(d_lin_maze,
                                                                       solution->x,
                                                                       solution->y,
                                                                       start.x,
                                                                       start.y,
                                                                       d_flag,
                                                                       5000,
                                                                       sizes_array[k]);

            cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


            for (int w = 0; w < max_iter; w++){
                cudaMemcpy(d_x_array, h_x_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_y_array, h_y_array, n_part[i] * sizeof(int), cudaMemcpyHostToDevice);
                cudaMemcpy(d_paths_array, h_paths_array, (MAX_SIZE_PATH + 2) * n_part[i] * sizeof(int), cudaMemcpyHostToDevice);

                cudaEvent_t t_start1, t_stop1;
                cudaEventCreate(&t_start1);
                cudaEventCreate(&t_stop1);

                cudaEventRecord(t_start1);

                while (*h_flag != 1){
                    n_steps++;
                    OneStepMazeWithPath<<< (n_part[i] + block_dim - 1) / block_dim, block_dim >>>(d_lin_maze,
                                                                                          start.x,
                                                                                          start.y,
                                                                                          d_flag,
                                                                                          n_part[i],
                                                                                          d_x_array,
                                                                                          d_y_array,
                                                                                          sizes_array[k],
                                                                                          d_paths_array);
                    cudaDeviceSynchronize();
                    cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

                }



                cudaEventRecord(t_stop1);
                cudaEventSynchronize(t_stop1);
                milliseconds = 0.0;
                cudaEventElapsedTime(&milliseconds, t_start1, t_stop1);
                times[k][i] += milliseconds;

                *h_flag = 0;
                cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

                //printf("%f\n", milliseconds);

                cudaEventDestroy(t_start1);
                cudaEventDestroy(t_stop1);
            }

            times[k][i] = times[k][i] / (float) max_iter;
            printf("%f, ", times[k][i]);

            n_steps = 0;
            free(h_x_array);
            free(h_y_array);
            free(h_paths_array);

            cudaFree(d_x_array);
            cudaFree(d_y_array);
            cudaFree(d_paths_array);
        }
        printf("\n");

        for (int i = 0; i < sizes_array[k]; ++i){
            free(maze[i]);
        }

        free(maze);
        free(h_lin_maze);
        free(solution);

        cudaFree(d_lin_maze);
        //printf("* - ");
    }
    //printf("|\n");

    free(h_flag);
    cudaFree(d_flag);

    return times;
}



int main() {

    // SINGLE RUN
    // #################################################################################################################
    /*
    int n_steps = 0;
    float time_seq, time_par;

    Point start;
    start.x = 0; start.y = 3;

    Point *solution;
    solution = (Point*) malloc(sizeof (Point));
    if (solution == nullptr) return 1;

    // Time variables
    float milliseconds = 0;
    cudaEvent_t t_start, t_stop;

    cudaEventCreate(&t_start);
    cudaEventCreate(&t_stop);



    // Maze matrix
    char **maze = performance_maze_init(start, solution, XDIM, YDIM);
    //print_maze(maze, XDIM, YDIM);
    // Linear maze
    short *h_lin_maze = return_lin_maze(maze, XDIM, YDIM);
    short *d_lin_maze;

    int *h_paths_array;
    h_paths_array = (int*) malloc((MAX_SIZE_PATH + 2) * NPART * sizeof(int));

    for (int i = 0; i < (MAX_SIZE_PATH + 2) * NPART; i++) {
        if (i % (MAX_SIZE_PATH + 2) == 0){
            h_paths_array[i] = (int) 0;
            h_paths_array[i + 1] = (int) 0;
            i += 1;
        }

        else h_paths_array[i] = (int) CONTROL_NUMBER;
    }




    int *d_paths_array;
    cudaMalloc((void**) &d_paths_array, (MAX_SIZE_PATH + 2) * NPART * sizeof(int));
    cudaMemcpy(d_paths_array, h_paths_array, (MAX_SIZE_PATH + 2) * NPART * sizeof(int), cudaMemcpyHostToDevice);



    // Host and device flags
    int *h_flag, *d_flag;
    h_flag = (int*) malloc(sizeof(int));
    *h_flag = 0;

    int *h_x_array = (int*) malloc(NPART * sizeof(int));
    int *h_y_array = (int*) malloc(NPART * sizeof(int));
    int *d_x_array, *d_y_array;


    for (int i = 0; i < NPART; i++) {
        h_x_array[i] = solution->x;
        h_y_array[i] = solution->y;
    }



    cudaMalloc((void**) &d_x_array, NPART * sizeof(int));
    cudaMalloc((void**) &d_y_array, NPART * sizeof(int));

    cudaMalloc((void**) &d_flag, sizeof(int));
    cudaMalloc((void**) &d_lin_maze, XDIM * YDIM * 11 * sizeof(short));


    cudaMemcpy(d_lin_maze, h_lin_maze, XDIM * YDIM * 11 * sizeof(short), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_x_array, h_x_array, NPART * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_array, h_y_array, NPART * sizeof(int), cudaMemcpyHostToDevice);


    // Warm-up
    RandSolver<<<(5000 + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM>>>(d_lin_maze,
                                                                solution->x,
                                                                solution->y,start.x,
                                                                start.y,
                                                                d_flag,
                                                                5000,
                                                                XDIM);

    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


    printf("N PARTICLES %d\n", NPART);
    printf("SIZE: %dX%d\n", XDIM, YDIM);
    cudaEventRecord(t_start);


    // NO PATHS
    // #################################################################################################################
    while(*h_flag != 1) {

        OneStepMaze<<< (NPART + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >>>(d_lin_maze,
                                                                       start.x,
                                                                       start.y,
                                                                       d_flag,
                                                                       NPART,
                                                                       d_x_array,
                                                                       d_y_array,
                                                                       XDIM);

        cudaDeviceSynchronize();
        cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

        n_steps++;
    }

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&milliseconds, t_start, t_stop);

    *h_flag = 0;
    cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice);


    time_par = milliseconds;

    printf("PAR SOLUTION (NO PATHS) FOUND IN    %f ms\n", time_par);




    // WITH PATHS
    // #################################################################################################################

    cudaMemcpy(d_x_array, h_x_array, NPART * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_array, h_y_array, NPART * sizeof(int), cudaMemcpyHostToDevice);
    n_steps = 0;
    cudaEventRecord(t_start);
    while(*h_flag != 1) {
        //printf("%d\n", n_steps);
        OneStepMazeWithPath<<< (NPART + BLOCKDIM - 1) / BLOCKDIM, BLOCKDIM >>>(d_lin_maze,
                                                                       start.x,
                                                                       start.y,
                                                                       d_flag,
                                                                       NPART,
                                                                       d_x_array,
                                                                       d_y_array,
                                                                       XDIM,
                                                                       d_paths_array);

        cudaDeviceSynchronize();
        cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
        n_steps++;
    }

    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&milliseconds, t_start, t_stop);


    cudaMemcpy(h_paths_array, d_paths_array, (MAX_SIZE_PATH + 2) * NPART * sizeof(int), cudaMemcpyDeviceToHost);

    time_par = milliseconds;
    printf("PAR SOLUTION (WITH PATHS) FOUND IN    %f ms\n", time_par);

    //int is_sol;

    //for (int i = 0; i < NPART * (MAX_SIZE_PATH + 2) ; i += (MAX_SIZE_PATH + 2)) {
    //    n_steps = h_paths_array[i];
    //    is_sol = h_paths_array[i + 1];
    //
    //    if (is_sol == 1){
    //        printf("SOLUTION FOUND IN %d STEPS\n", n_steps);
    //        if (n_steps < MAX_SIZE_PATH){
    //            for (int j = 0; j < n_steps; j++){
    //                printf("%d, ", h_paths_array[i + j]);
    //            }
    //            printf("\n");
    //        }
    //    }
    //}


    //printf("\n");
    //for (int i = 0; i < (MAX_SIZE_PATH + 2) * NPART; i++){
    //    printf("%hd ", h_paths_array[i]);
    //    if ((i+1) % (MAX_SIZE_PATH + 2) == 0) printf("\n");
    //}



    // SEQ VERSION
    // #################################################################################################################

    cudaEventRecord(t_start);
    rand_solver_cpu(h_lin_maze, XDIM, YDIM, start.x, start.y, NPART, h_x_array, h_y_array, 2 * XDIM);
    cudaEventRecord(t_stop);
    cudaEventSynchronize(t_stop);
    cudaEventElapsedTime(&milliseconds, t_start, t_stop);

    time_seq = milliseconds;

    printf("SEQ SOLUTION FOUND IN    %f ms\n\n", time_seq);
    printf("SPEEDUP:    %.3f", time_seq / time_par);


    //  FREE MEMORY

    for (int i = 0; i < YDIM; ++i){
        free(maze[i]);
    }

    free(maze);
    free(solution);
    free(h_flag);
    free(h_lin_maze);
    free(h_x_array);
    free(h_y_array);
    free(h_paths_array);

    // CUDA FREE MEMORY
    cudaFree(d_lin_maze);
    cudaFree(d_flag);
    cudaFree(d_x_array);
    cudaFree(d_y_array);
    cudaEventDestroy(t_start);
    cudaEventDestroy(t_stop);
    cudaFree(d_paths_array);
    */



    // SPEED UP CURVES
    // #################################################################################################################


    // MAZE DIM: 500X500

    int n_part[5] = {10000, 50000, 100000, 200000, 500000};
    int block_dim_array[7] = {16, 32, 64, 128, 256, 512, 1024};
    float **times = collect_speedup(XDIM, YDIM, n_part, 5, block_dim_array, 7);


    for (int i = 0; i < 5; i++){
        for (int j = 0; j < 7; j++) printf("%f, ", times[i][j]);

        printf("\n");
        free(times[i]);
    }
    free(times);



    // TIMES FOR DIFFERENT SIZE
    // #################################################################################################################
    /*
    int n_part[6] = {300, 500, 1000, 5000, 10000, 50000};
    int size_dim[5] = {50, 100, 200, 300, 400};

    float **times = collect_times(n_part, 6, size_dim, 5, 20);
    //float **times = collect_times_with_path(n_part, 6, size_dim, 5, 20);


    for (int i = 0; i < 1; i++){
        //for (int j = 0; j < 8; j++) printf("%f, ", times[i][j]);
        //printf("\n");
        free(times[i]);
    }
    free(times);
    */
}
