#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double* init_matrix(int n) {
    double* mat = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n * n; i++) {
        mat[i] = (double)rand() / RAND_MAX;
    }
    return mat;
}

// sequential version (mode 0)
void matmul_seq(int n, double* A, double* B, double* C) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// naive openmp parallel version (mode 1)
void matmul_parallel(int n, double* A, double* B, double* C) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i*n + k] * B[k*n + j];
            }
            C[i*n + j] = sum;
        }
    }
}

// blocked version (omde 2)
void matmul_blocked(int n, double* A, double* B, double* C, int block) {
    #pragma omp parallel for collapse(2)
    for (int a = 0; a < n; a += block) {
        for (int b = 0; b < n; b += block) {
            for (int c = 0; c < n; c += block) {
                for (int i = a; i < a + block && i < n; i++) {
                    for (int k = c; k < c + block && k < n; k++) {
                        double a_ik = A[i*n + k];
                        for (int j = b; j < b + block && j < n; j++) {
                            C[i*n + j] += a_ik * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

// guided blocking version (mode 3)
void matmul_blocked_guided(int n, double* A, double* B, double* C) {
    int block;
    if (n <= 1024) block = 32;
    else if (n <= 4096) block = 64;
    else if (n <= 8192) block = 128;
    else block = 256;

    #pragma omp parallel for collapse(2) schedule(guided)
    for (int a = 0; a < n; a += block) {
        for (int b = 0; b < n; b += block) {
            for (int c = 0; c < n; c += block) {
                for (int i = a; i < a + block && i < n; i++) {
                    for (int k = c; k < c + block && k < n; k++) {
                        double a_ik = A[i*n + k];
                        for (int j = b; j < b + block && j < n; j++) {
                            C[i*n + j] += a_ik * B[k*n + j];
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: %s <matrix_size> <num_threads> <mode>\n", argv[0]);
        printf("Modes:\n");
        printf("  0 = Sequential (no OpenMP)\n");
        printf("  1 = Naive Parallel (OpenMP)\n");
        printf("  2 = Memory-Optimized (Blocked/Tiled)\n");
        printf("  3 = Guided + Adaptive Blocking (Advanced)\n");
        return 1;
    }
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    int mode = atoi(argv[3]);

    int block_size = 64;

    double* A = (double*)malloc(n * n * sizeof(double));
    double* B = (double*)malloc(n * n * sizeof(double));
    double* C = (double*)calloc(n * n, sizeof(double));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }

    double start = omp_get_wtime();

    if (mode == 0) {
        matmul_seq(n, A, B, C);
    } else if (mode == 1) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
        matmul_parallel(n, A, B, C);
    } else if (mode == 2) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
        matmul_blocked(n, A, B, C, block_size);
    }
    else if (mode == 3) {
        if (num_threads > 0) omp_set_num_threads(num_threads);
        matmul_blocked_guided(n, A, B, C);
    } else {
    printf("Invalid mode. Use 0, 1, 2, or 3.\n");
    free(A); free(B); free(C);
    return 1;
    }

    double end = omp_get_wtime();

    printf("Matrix: %dx%d | Threads: %d | Mode: %d | Time: %.6f sec\n",
           n, n, num_threads, mode, end - start);

    free(A);
    free(B);
    free(C);
    return 0;
}