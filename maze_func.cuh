//
// Created by angel on 29/03/2024.
//

#ifndef MAIN_CU_MAZE_FUNC_CUH
#define MAIN_CU_MAZE_FUNC_CUH
#include "vector"
#include "curand_kernel.h"

#define WAY '.'
#define WALL '+'
#define EMPTY 'E'
#define START 'X'

#define N 1000000

#define XMAX 100
#define YMAX 100


#define NITER 100

typedef struct Point{
    int x;
    int y;

    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
} Point;

typedef struct {
    std::vector <int> x;
    std::vector <int> y;
    //char moves[N][50];
} Particles;

typedef struct {
    int n_ways;
    int x[4];
    int y[4];
    //char moves[4];
} Adjacents;




char **maze_init(Point start, Point *solution, int x_max, int y_max);
void print_maze(char **maze, int x_max, int y_max);
int find_ways(char **maze, int x_max, int y_max, Adjacents *adjac, int x, int y);
float rand_solver_cpu(const short *h_lin_maze, int x_dim, int y_dim, int x_ext, int y_ext, int n, int *x_array, int *y_array, int max_steps);
char **performance_maze_init(Point start, Point *solution, int x_max, int y_max);


#endif //MAIN_CU_MAZE_FUNC_CUH
