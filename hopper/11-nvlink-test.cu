#include "multi-gpu.cuh"

int main(int argc, char **argv) {

    unsigned int devices_count {};

    nvmlInit ();
    nvmlDeviceGetCount (&devices_count);

    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex (0, &device);

    nvmlFieldValue_t field;
    field.scopeId = 0;
    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int initial_tx = field.value.ullVal;

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int initial_rx = field.value.ullVal;

    // code to measure

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int final_tx = field.value.ullVal;

    field.fieldId = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    nvmlDeviceGetFieldValues (device, 1, &field);
    const unsigned long long int final_rx = field.value.ullVal;

    const unsigned int rx = final_rx - initial_rx;
    const unsigned int tx = final_tx - initial_tx;
}