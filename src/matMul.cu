#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// 1. Kernel Naïve: Acceso directo a memoria global
__global__ void matMulNaive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 2. Kernel optimizado mediante Tiling y Shared Memory
__global__ void matMulTiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE_SIZE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Multiplicación en CPU para validar cálculos
void matMulCPU(const float *A, const float *B, float *C, int N) {
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

void runBenchmark(int N) {
    printf("\n=== Ejecutando Benchmark para N = %d ===\n", N);
    size_t bytes = N * N * sizeof(float);

    float *h_A   = (float*)malloc(bytes);
    float *h_B   = (float*)malloc(bytes);
    float *h_C   = (float*)malloc(bytes);
    float *h_C_g = (float*)malloc(bytes); // Corregido el nombre para coincidir abajo

    for (int i = 0; i < N * N; i++) {
        h_A[i] = (float)(rand() % 10) / 10.0f;
        h_B[i] = (float)(rand() % 10) / 10.0f;
    }

    if(N <= 512) {
        clock_t t0 = clock();
        matMulCPU(h_A, h_B, h_C, N);
        double cpu_ms = ((double)(clock() - t0) / CLOCKS_PER_SEC) * 1000.0;
        printf("Tiempo CPU: %.2f ms\n", cpu_ms);
    } else {
        printf("Tiempo CPU: Omitido (N grande)\n");
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- Ejecución Naïve ---
    cudaEventRecord(start);
    matMulNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float naive_ms = 0;
    cudaEventElapsedTime(&naive_ms, start, stop);
    printf("GPU Kernel (Naive): %.2f ms\n", naive_ms);

    // --- Ejecución Tiled con Shared Memory ---
    cudaEventRecord(start);
    matMulTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float tiled_ms = 0;
    cudaEventElapsedTime(&tiled_ms, start, stop);
    printf("GPU Kernel (Tiled Shared Memory): %.2f ms\n", tiled_ms);
    
    if (naive_ms > 0) {
        printf("Speedup de Tiling frente a Naive: %.2fx\n", naive_ms / tiled_ms);
    }

    // Validación de precisión (en N=512)
    if(N <= 512) {
        cudaMemcpy(h_C_g, d_C, bytes, cudaMemcpyDeviceToHost); // Usando la variable correcta h_C_g
        int errors = 0;
        for (int i = 0; i < N*N; i++) {
            if (fabs(h_C_g[i] - h_C[i]) > 1e-3f) errors++;
        }
        printf("Errores vs CPU: %d\n", errors);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_g);
}

int main() {
    cudaSetDevice(0); // Mantenemos esto para asegurar que use tu RTX 2050 en Windows
    srand(time(NULL));

    runBenchmark(512);
    runBenchmark(1024);
    return 0;
}