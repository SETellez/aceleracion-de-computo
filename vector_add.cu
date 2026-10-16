#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t error = (call);                                         \
        if (error != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                         __FILE__, __LINE__, cudaGetErrorString(error));     \
            return 1;                                                        \
        }                                                                    \
    } while (false)

constexpr int kElements = 1 << 20;
constexpr int kThreadsPerBlock = 256;

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const std::size_t bytes = kElements * sizeof(float);
    const int blocks = (kElements + kThreadsPerBlock - 1) / kThreadsPerBlock;

    float* hostA = static_cast<float*>(std::malloc(bytes));
    float* hostB = static_cast<float*>(std::malloc(bytes));
    float* hostC = static_cast<float*>(std::malloc(bytes));
    if (hostA == nullptr || hostB == nullptr || hostC == nullptr) {
        std::fprintf(stderr, "No se pudo reservar memoria en el host.\n");
        std::free(hostA);
        std::free(hostB);
        std::free(hostC);
        return 1;
    }

    for (int index = 0; index < kElements; ++index) {
        hostA[index] = std::sin(static_cast<float>(index));
        hostB[index] = std::cos(static_cast<float>(index));
    }

    float* deviceA = nullptr;
    float* deviceB = nullptr;
    float* deviceC = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, hostA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, hostB, bytes, cudaMemcpyHostToDevice));

    vectorAdd<<<blocks, kThreadsPerBlock>>>(deviceA, deviceB, deviceC,
                                             kElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hostC, deviceC, bytes, cudaMemcpyDeviceToHost));

    int errors = 0;
    constexpr float tolerance = 1e-5f;
    for (int index = 0; index < kElements; ++index) {
        const float expected = hostA[index] + hostB[index];
        if (std::fabs(hostC[index] - expected) > tolerance) {
            ++errors;
        }
    }

    std::printf("Elementos: %d | bloques: %d | hilos por bloque: %d\n",
                kElements, blocks, kThreadsPerBlock);
    std::printf("c[0] = %.6f | esperado = %.6f\n", hostC[0],
                hostA[0] + hostB[0]);
    std::printf("c[%d] = %.6f | esperado = %.6f\n", kElements - 1,
                hostC[kElements - 1], hostA[kElements - 1] + hostB[kElements - 1]);
    std::printf("Verificacion: %s (%d errores)\n",
                errors == 0 ? "OK" : "FALLO", errors);

    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));
    std::free(hostA);
    std::free(hostB);
    std::free(hostC);
    return errors == 0 ? 0 : 1;
}