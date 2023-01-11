#include <iostream>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048

int size, rank;
double start_time, end_time;
int a[N * N], b[N * N], c[N * N];

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int* a_row = new int[N / size * N];
    int* c_row = new int[N / size * N];

    start_time = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                a[i * N + j] = i + j;
                b[i * N + j] = i + j;
            }
        }

        MPI_Scatter(a, N / size * N, MPI_INT, a_row, N / size * N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N / size; i++) {
            for (int j = 0; j < N; j++) {
                c[i * N + j] = 0;
                for (int k = 0; k < N; k++) {
                    c[i * N + j] += a_row[i * N + k] * b[j + k * N];
                }
            }
        }

        MPI_Gather(c_row, N / size * N, MPI_INT, c, N / size * N, MPI_INT, 0, MPI_COMM_WORLD);

        int last_row = N % size;
        if (last_row != 0) {
            for (int i = N - last_row; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    c[i * N + j] = 0;
                    for (int k = 0; k < N; k++) {
                        c[i * N + j] += a[i * N + k] * b[j + k * N];
                    }
                }
            }
        }

        end_time = MPI_Wtime();

        std::cout << end_time - start_time << std::endl;
    }
    else {
        MPI_Scatter(a, N / size * N, MPI_INT, a_row, N / size * N, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, N * N, MPI_INT, 0, MPI_COMM_WORLD);

        for (int i = 0; i < N / size; i++) {
            for (int j = 0; j < N; j++) {
                c_row[i * N + j] = 0;
                for (int k = 0; k < N; k++) {
                    c_row[i * N + j] += a_row[i * N + k] * b[j + k * N];
                }
            }
        }

        MPI_Gather(c_row, N / size * N, MPI_INT, c, N / size * N, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}