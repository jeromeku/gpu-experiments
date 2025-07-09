#include "multi-gpu.cuh"

int main() {
    // Constants 
    constexpr int n = 2048;

    printf("Matrix dimension: %d\n", n);
    printf("Allocating matrices...\n");

    // Allocate host matrices
    float *a = (float*)malloc(n * n * sizeof(float));
    float *b = (float*)malloc(n * n * sizeof(float));
    float *c = (float*)malloc(n * n * sizeof(float));

    // Initialize matrices with random values
    for (int i = 0; i < n * n; ++i) {
        a[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
        b[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
    memset(c, 0, n * n * sizeof(float));

    printf("Computing matrix multiplication...\n");

    // Compute matrix multiplication C = A * B
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    printf("Matrix multiplication completed\n");

    // Cleanup
    free(a);
    free(b);
    free(c);

    return 0;
}