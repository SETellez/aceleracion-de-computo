#include <cuda_runtime.h>

#include <cstdio>

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t error = (call);                                         \
        if (error != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(error));     \
            return 1;                                                        \
        }                                                                    \
    } while (false)

__global__ void helloKernel() {
    std::printf("Hello World desde CUDA | bloque %d | hilo %d\n",
                blockIdx.x, threadIdx.x);
}

int main() {
    cudaDeviceProp device{};
    CUDA_CHECK(cudaGetDeviceProperties(&device, 0));

    std::printf("GPU activa: %s\n", device.name);
    helloKernel<<<1, 1>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Ejecucion finalizada correctamente.\n");
    return 0;
}