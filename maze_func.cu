//
// Created by angel on 29/03/2024.
//

#include "maze_func.cuh"
#include "cmath"
#include "ctime"
#include "cstdio"


int return_empty_adjacent(char **maze, Point *adjacent, Point current, int x_max, int y_max){

    Point vector[4];
    int counter = 0;

    if (current.x + 1 < x_max){
        if (maze[current.y][current.x + 1] == EMPTY){
            vector[counter].x = current.x + 1;
            vector[counter].y = current.y;
            counter++;
        }
    }

    if (current.x - 1 > 0){
        if (maze[current.y][current.x - 1] == EMPTY){
            vector[counter].x = current.x - 1;
            vector[counter].y = current.y;
            counter++;
        }
    }

    if (current.y + 1 < y_max){
        if (maze[current.y + 1][current.x] == EMPTY){
            vector[counter].x = current.x;
            vector[counter].y = current.y + 1;
            counter++;
        }
    }

    if (current.y - 1 > 0){
        if (maze[current.y - 1][current.x] == EMPTY){
            vector[counter].x = current.x;
            vector[counter].y = current.y - 1;
            counter++;
        }
    }

    if (counter == 0) return 0;

    for (int i = 0; i < counter; ++i) {
        adjacent[i].x = vector[i].x;
        adjacent[i].y = vector[i].y;
    }

    return counter;
}

int return_adjacent_ways(char **maze, Point current, int x_max, int y_max){
    int counter = 0;

    if (current.x + 1 < x_max){
        if (maze[current.y][current.x + 1] == WAY) counter++;
    }

    if (current.x - 1 > 0){
        if (maze[current.y][current.x - 1] == WAY) counter++;
    }

    if (current.y + 1 < y_max){
        if (maze[current.y + 1][current.x] == WAY) counter++;
    }

    if (current.y - 1 > 0){
        if (maze[current.y - 1][current.x] == WAY) counter++;
    }

    return counter;
}

void wall_filler(char **maze, Point current, int x_max, int y_max){
    int n_ways;
    n_ways = return_adjacent_ways(maze, current, x_max, y_max);
    //printf("(%d, %d)", current.x, current.y);
    //printf(" COUNTS: %d\n", n_ways);
    if (n_ways > 1) maze[current.y][current.x] = WALL;
}

Point path_tracker(char **maze, Point start, int x_max, int y_max, int is_solution){

    int tot_steps;
    if(is_solution == 1){
        tot_steps = (int) (2.5 * x_max);
    }
    else {
        tot_steps = (int) ( sqrt(pow(x_max, 2) + pow(y_max, 2)) * 1.75 );
    }

    //printf("%d\n", tot_steps);
    int step = 0, len, choice;
    Point adjacent[4], current;

    srand(time(nullptr));
    len = return_empty_adjacent(maze, adjacent, start, x_max, y_max);

    while(step < tot_steps){

        if (len != 0){

            choice = rand() % len;

            maze[adjacent[choice].y][adjacent[choice].x] = WAY;
            current = adjacent[choice];
            //printf("POINT: (%d, %d)\nADJACENTS\n", current.x, current.y);
            len = return_empty_adjacent(maze, adjacent, current, x_max, y_max);

            for (int i = 0; i < len; ++i) {
                wall_filler(maze, adjacent[i], x_max, y_max);
            }
            //printf("______________\n\n");

            len = return_empty_adjacent(maze, adjacent, current, x_max, y_max);

        }

        else break;

        step++;
    }

    if (is_solution == 1) maze[current.y][current.x] = START;

    return current;
}

char **maze_init(Point start, Point *solution, int const x_max, int const y_max){

    int counter, len;
    std::vector <Point> vec(x_max * y_max);
    Point neigh[4], current;
    char **maze = (char**) malloc (y_max * sizeof(char*));
    if (maze == nullptr) return nullptr;

    for (int i = 0; i < y_max; ++i) {
        maze[i] = (char*) malloc (x_max * sizeof (char));
        if (maze[i] == nullptr) return nullptr;
    }



    srand(time(nullptr));

    for(int i = 0; i < y_max; i++){
        for (int j = 0; j < x_max; ++j) {
            if(i == 0 || i == y_max -1 || j == 0 || j == x_max - 1){
                maze[i][j] = (i == start.y && j == start.x) ? WAY: WALL;
            }
            else maze[i][j] = EMPTY;
        }
    }

    *solution = path_tracker(maze, start, x_max, y_max, 1);

    do{
        counter = 0;
        for (int i = 0; i < y_max; ++i) {
            for (int j = 0; j < x_max; ++j) {
                if (maze[i][j]  == WAY){
                    current.x = j; current.y = i;
                    len = return_empty_adjacent(maze, neigh, current, x_max, y_max);
                    if (len != 0){
                        vec[counter] = current;
                        counter++;
                    }
                }
            }
        }

        if (counter != 0){
            current = vec[rand() % counter];
            path_tracker(maze, current, x_max, y_max, 0);
        }

    }
    while (counter != 0);

    for (int i = 0; i < y_max; ++i) {
        for (int j = 0; j < x_max; ++j) {
            if (maze[i][j] == EMPTY) maze[i][j] = WALL;
        }
    }

    return maze;
}

void print_maze(char **maze, int x_max, int y_max){
    for (int i = 0; i < y_max; ++i) {
        for (int j = 0; j < x_max; ++j) {
            printf("%c", maze[i][j]);
        }
        printf("\n");
    }
}

int find_ways(char **maze, int x_max, int y_max, Adjacents *adjac, int x, int y){

    int counter = 0;

    if (x + 1 < x_max){
        if (maze[y][x + 1] == WAY || maze[y][x + 1] == START){
            adjac->x[counter] = x + 1;
            adjac->y[counter] = y;
            //adjac->moves[counter] = 'R';
            counter++;
        }
    }

    if (x - 1 >= 0){
        if (maze[y][x - 1] == WAY || maze[y][x - 1] == START){
            adjac->x[counter] = x - 1;
            adjac->y[counter] = y;
            //adjac->moves[counter] = 'L';
            counter++;
        }
    }

    if (y + 1 < y_max){
        if (maze[y + 1][x] == WAY || maze[y + 1][x] == START){
            adjac->x[counter] = x;
            adjac->y[counter] = y + 1;
            //adjac->moves[counter] = 'D';
            counter++;
        }
    }

    if (y - 1 >= 0){
        if (maze[y - 1][x] == WAY || maze[y - 1][x] == START){
            adjac->x[counter] = x;
            adjac->y[counter] = y - 1;
            //adjac->moves[counter] = 'U';
            counter++;
        }
    }

    return counter;
}

int cpu_random_solver(char **maze, int x_max, int y_max, Particles particles, int x_ext, int y_ext){
    int n_ways, rand_choice, n_steps = -1;
    Adjacents *adjac;
    adjac = (Adjacents*) malloc(sizeof(Adjacents));

    //n_steps != NITER
    //true
    while(true){
        n_steps += 1;
        for (int i = 0; i < N; ++i) {
            n_ways = find_ways(maze, x_max, y_max, adjac, particles.x[i], particles.y[i]);
            rand_choice = rand() % n_ways;
            particles.x[i] = adjac->x[rand_choice];
            particles.y[i] = adjac->y[rand_choice];
            //particles.moves[i][n_steps] = adjac->moves[rand_choice];

            //particles.x[i] == x_ext && particles.y[i] == y_ext
            //n_steps == NITER
            if (particles.x[i] == x_ext && particles.y[i] == y_ext){
                free(adjac);
                /*
                printf("PATH:\n");
                for (int j = 0; j < n_steps; ++j) {
                    printf("%c ", particles.moves[i][j]);
                }
                */
                return n_steps + 1;
            }
        }
    }
}

float rand_solver_cpu(const short *h_lin_maze, int x_dim, int y_dim, int x_ext, int y_ext, int n, int *x_array, int *y_array, int max_steps){

    int rand_choice, n_steps = 0, temp, flag = 0;
    clock_t  start, end;

    start = clock();
    while(/*flag == 0 &&*/ n_steps < 1){
        n_steps++;
        for (int i = 0; i < n; ++i){
            for (int j = 0; j < max_steps; j++){
                rand_choice = rand() % (h_lin_maze[11 * y_array[i] * x_dim + 11 * x_array[i] + 2] ) + 1;
                temp = h_lin_maze[11 * y_array[i] * x_dim + 11 * x_array[i] + 2 + 2 * rand_choice - 1];
                y_array[i] = h_lin_maze[11 * y_array[i] * x_dim + 11 * x_array[i] + 2 + 2 * rand_choice];
                x_array[i] = temp;

                //x_vector[i] == x_ext && y_vector[i] == y_ext

                if (x_array[i] == x_ext && y_array[i] == y_ext){
                    flag = 1;
                    end = clock();
                    //return (float) (end - start)/ CLOCKS_PER_SEC;
                }
            }
        }
    }
    end = clock();
    return (float) (end - start)/ CLOCKS_PER_SEC;
}






int distance_evaluate(Point start, Point stop, int dist, int dev){
    //printf("%d\n", dist);
    int man_dist = abs(start.x - stop.x) + abs(start.y - stop.y);
    if(man_dist >= dist - dev && man_dist <= dist + dev) return 1;
    return 0;
}


Point performance_path_tracker(char **maze, Point start, int x_max, int y_max, std::vector<Point>& vec, int *start_found, Point init, Point *solution){

    int tot_steps = (int) ( sqrt(pow(x_max, 2) + pow(y_max, 2)) * 1.75 );

    //printf("%d\n", tot_steps);
    int step = 0, len, choice;
    Point adjacent[4], current;
    current = start;

    srand(time(nullptr));
    len = return_empty_adjacent(maze, adjacent, start, x_max, y_max);

    while(step < tot_steps){

        if (len != 0){
            if (len == 1){
                vec.erase(std::remove(vec.begin(), vec.end(), current), vec.end());
            }


            choice = rand() % len;

            if (*start_found == 0){
                if (distance_evaluate(init, adjacent[choice],1.75 * x_max , 5) == 1){
                    *start_found = 1;
                    maze[adjacent[choice].y][adjacent[choice].x] = START;
                    *solution = adjacent[choice];
                    //printf("DISTANCE: %d\n", abs(init.x - adjacent[choice].x) + abs(init.y - adjacent[choice].y));
                    break;
                }
            }

            maze[adjacent[choice].y][adjacent[choice].x] = WAY;
            current = adjacent[choice];
            //printf("POINT: (%d, %d)\nADJACENTS\n", current.x, current.y);
            len = return_empty_adjacent(maze, adjacent, current, x_max, y_max);

            for (int i = 0; i < len; ++i) {
                wall_filler(maze, adjacent[i], x_max, y_max);
            }
            //printf("______________\n\n");

            vec.push_back(current);
            len = return_empty_adjacent(maze, adjacent, current, x_max, y_max);

        }

        else{
            vec.erase(std::remove(vec.begin(), vec.end(), current), vec.end());
            break;
        }

        step++;
    }

    return current;
}

char **performance_maze_init(Point start, Point *solution, int x_max, int y_max){

    int counter = 0, len;
    std::vector <Point> vec(1);
    vec[0] = start;

    int *start_found;
    start_found = (int*) malloc(sizeof(int));
    *start_found = 0;

    Point neigh[4], current = start;
    char **maze = (char**) malloc (y_max * sizeof(char*));
    if (maze == nullptr) return nullptr;

    srand(time(nullptr));

    for (int i = 0; i < y_max; ++i) {
        maze[i] = (char*) malloc (x_max * sizeof (char));
        if (maze[i] == nullptr) return nullptr;

        for (int j = 0; j < x_max; ++j) {
            if(i == 0 || i == y_max -1 || j == 0 || j == x_max - 1){
                maze[i][j] = (i == start.y && j == start.x) ? WAY : WALL;
            }
            else{
                maze[i][j] = EMPTY;
            }
        }
    }

    while (vec.size() != 0) {
        current = vec[rand() % (int) vec.size()];
        performance_path_tracker(maze, current, x_max, y_max, vec, start_found, start, solution);
    }

    free(start_found);
    return maze;
}




