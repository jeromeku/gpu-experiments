/*
    A simple implementation of ring reduction, just so I can personally understand it better
*/

#include "multi-gpu.cuh"

constexpr int NUM_RANKS = 4;
constexpr int NUM_ELEMS = 4;

// No real threads or processes
static float data[NUM_RANKS][NUM_ELEMS];
static float rx_buffer[NUM_RANKS]; // to simulate "receiving" of data from other rank

void ring_reduce_share_reduce_step1(int rank, int iter) {
    // Send data to the next rank
    int next_rank = (rank + 1) % NUM_RANKS;
    int data_idx = (rank - iter + NUM_RANKS) % NUM_RANKS;

    rx_buffer[next_rank] = data[rank][data_idx]; // send data[p - iter] to process p + 1
}
void ring_reduce_share_reduce_step2(int rank, int iter) {
    // Use the received data from process p - 1 to perform reduce
    int data_idx = (rank - 1 - iter + NUM_RANKS) % NUM_RANKS;
    data[rank][data_idx] += rx_buffer[rank];
}

void ring_reduce_share_only_step(int rank, int iter) {
    int next_rank = (rank + 1) % NUM_RANKS;
    int data_idx = (rank + 1 - iter + NUM_RANKS) % NUM_RANKS;

    // no need for buffer here, just "send" directly
    data[next_rank][data_idx] = data[rank][data_idx];
}

int main() {
    // Data setup
    srand(static_cast<unsigned int>(time(nullptr))); // random seed
    printf("\nBefore reduction\n");
    for (int i = 0; i < NUM_RANKS; ++i) {
        printf("Rank %d: ", i);
        for (int j = 0; j < NUM_ELEMS; ++j) {
            data[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }

    float expected[NUM_ELEMS];
    printf("\nExpected result\n");
    for (int i = 0; i < NUM_ELEMS; ++i) {
        expected[i] = 0.0;
        for (int j = 0; j < NUM_RANKS; ++j) {
            expected[i] += data[j][i];
        }
        printf("%f ", expected[i]);
    }
    printf("\n");

    // Perform ring allreduce
    for (int iter = 0; iter < NUM_RANKS - 1; ++iter) { // note the total number of iters
        for (int rank = 0; rank < NUM_RANKS; ++rank) {
            ring_reduce_share_reduce_step1(rank, iter);
        }
        for (int rank = 0; rank < NUM_RANKS; ++rank) {
            ring_reduce_share_reduce_step2(rank, iter);
        }
        // printf("\nAfter iter %d of share-reduce phase\n", iter);
        // for (int i = 0; i < NUM_RANKS; ++i) {
        //     printf("Rank %d: ", i);
        //     for (int j = 0; j < NUM_ELEMS; ++j) {
        //         printf("%f ", data[i][j]);
        //     }
        //     printf("\n");
        // }
    }

    printf("\nAfter share-reduce phase\n");
    for (int i = 0; i < NUM_RANKS; ++i) {
        printf("Rank %d: ", i);
        for (int j = 0; j < NUM_ELEMS; ++j) {
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }

    for (int iter = 0; iter < NUM_RANKS - 1; ++iter) { // note the total number of iters
        for (int rank = 0; rank < NUM_RANKS; ++rank) {
            ring_reduce_share_only_step(rank, iter);
        }
    }

    printf("\nAfter share-only phase\n");
    for (int i = 0; i < NUM_RANKS; ++i) {
        printf("Rank %d: ", i);
        for (int j = 0; j < NUM_ELEMS; ++j) {
            printf("%f ", data[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // Verify results
    for (int rank = 0; rank < NUM_RANKS; ++rank) {
        for (int i = 0; i < NUM_ELEMS; ++i) {
            if (fabs(data[rank][i] - expected[i]) > 1e-5) {
                printf("ERROR: Rank %d, element %d, expected %f, got %f\n", rank, i, expected[i], data[rank][i]);
                return 1;
            }
        }
    }

    printf("SUCCESS\n");
    return 0;
}
