// Minimal NeuroPilot TFLite Shim benchmark for ANVIL operator profiling.
// Uses MediaTek's pre-installed libtflite_mtk.so via the Shim API header.
// Build: NDK CMake cross-compile, no extra .so needed.
// Usage: ./bench_neuropilot --graph=model.tflite [--accel=auto|cpu|neuron] [--num_runs=10]

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "NeuroPilotTFLiteShim.h"

static std::string FLAG_graph;
static std::string FLAG_accel = "auto";  // auto, cpu, neuron
static std::string FLAG_input_raw;   // optional: load input from raw file
static std::string FLAG_output_raw;  // optional: save output to raw file
static int FLAG_num_runs = 10;
static int FLAG_warmup = 3;

void parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg.find("--graph=") == 0) FLAG_graph = arg.substr(8);
        else if (arg.find("--accel=") == 0) FLAG_accel = arg.substr(8);
        else if (arg.find("--input_raw=") == 0) FLAG_input_raw = arg.substr(12);
        else if (arg.find("--output_raw=") == 0) FLAG_output_raw = arg.substr(13);
        else if (arg.find("--num_runs=") == 0) FLAG_num_runs = atoi(arg.substr(11).c_str());
        else if (arg.find("--warmup=") == 0) FLAG_warmup = atoi(arg.substr(9).c_str());
        else { fprintf(stderr, "Unknown: %s\n", argv[i]); exit(1); }
    }
    if (FLAG_graph.empty()) {
        fprintf(stderr, "Usage: %s --graph=model.tflite [--accel=auto|cpu|neuron] [--input_raw=in.raw] [--output_raw=out.raw] [--num_runs=10]\n", argv[0]);
        exit(1);
    }
}

int main(int argc, char** argv) {
    parse_args(argc, argv);

    printf("NeuroPilot Shim Benchmark\n");
    printf("  Model: %s\n", FLAG_graph.c_str());
    printf("  Accel: %s\n", FLAG_accel.c_str());
    printf("  Runs:  %d warmup + %d measured\n", FLAG_warmup, FLAG_num_runs);

    // Create options
    ANeuralNetworksTFLiteOptions* options = nullptr;
    int ret = ANeuralNetworksTFLiteOptions_create(&options);
    if (ret != 0 || !options) {
        fprintf(stderr, "ERROR: Failed to create options (%d)\n", ret);
        return 1;
    }

    // Set acceleration
    if (FLAG_accel == "neuron") {
        ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_NEURON);
        printf("  Mode:  NEURON (APU)\n");
    } else if (FLAG_accel == "cpu") {
        ANeuralNetworksTFLiteOptions_setAccelerationMode(options, NP_ACCELERATION_CPU);
        printf("  Mode:  CPU\n");
    } else {
        printf("  Mode:  AUTO\n");
    }

    // Set turbo boost for max performance
    ANeuralNetworksTFLiteOptions_setPreference(options, kTurboBoost);
    ANeuralNetworksTFLiteOptions_setBoostHint(options, 100);

    // Create TFLite instance with options (Adv = advanced, takes options)
    ANeuralNetworksTFLite* tflite = nullptr;
    ret = ANeuroPilotTFLiteWrapper_makeAdvTFLite(&tflite, FLAG_graph.c_str(), options);
    if (ret != 0 || !tflite) {
        fprintf(stderr, "ERROR: Failed to create TFLite instance (%d)\n", ret);
        ANeuralNetworksTFLiteOptions_free(options);
        return 1;
    }
    printf("  Model loaded OK\n");

    // Load input from raw file if specified
    if (!FLAG_input_raw.empty()) {
        FILE* f = fopen(FLAG_input_raw.c_str(), "rb");
        if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", FLAG_input_raw.c_str()); return 1; }
        fseek(f, 0, SEEK_END);
        size_t fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        std::vector<char> buf(fsize);
        fread(buf.data(), 1, fsize, f);
        fclose(f);
        ret = ANeuroPilotTFLiteWrapper_setInputTensorData(tflite, 0, buf.data(), fsize);
        if (ret != 0) { fprintf(stderr, "ERROR: setInputTensorData failed (%d)\n", ret); return 1; }
        printf("  Input loaded: %s (%zu bytes)\n", FLAG_input_raw.c_str(), fsize);
    }

    // Warmup
    printf("Warmup...\n");
    for (int i = 0; i < FLAG_warmup; i++) {
        ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
        if (ret != 0) {
            fprintf(stderr, "ERROR: Warmup invoke failed (%d)\n", ret);
            break;
        }
    }

    // Timed runs
    printf("Benchmarking...\n");
    std::vector<double> times;
    times.reserve(FLAG_num_runs);

    for (int i = 0; i < FLAG_num_runs; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ret = ANeuroPilotTFLiteWrapper_invoke(tflite);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (ret != 0) {
            fprintf(stderr, "ERROR: Run %d failed (%d)\n", i, ret);
            continue;
        }
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times.push_back(ms);
    }

    // Save output if requested
    if (!FLAG_output_raw.empty()) {
        // Get output size: 1 * 3 * H * W * sizeof(float) for FP32 model
        // For INT8 model the output may be quantized - just save raw bytes
        size_t out_size = 1 * 3 * 1080 * 1920 * 4;  // default FP32 1080p
        // Try a generous buffer
        std::vector<char> out_buf(out_size);
        ret = ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite, 0, out_buf.data(), out_size);
        if (ret != 0) {
            fprintf(stderr, "WARNING: getOutputTensorData failed (%d), trying smaller sizes\n", ret);
            // Try INT8 size (1 byte per element)
            out_size = 1 * 3 * 1080 * 1920;
            out_buf.resize(out_size);
            ret = ANeuroPilotTFLiteWrapper_getOutputTensorData(tflite, 0, out_buf.data(), out_size);
        }
        if (ret == 0) {
            FILE* f = fopen(FLAG_output_raw.c_str(), "wb");
            if (f) {
                fwrite(out_buf.data(), 1, out_size, f);
                fclose(f);
                printf("  Output saved: %s (%zu bytes)\n", FLAG_output_raw.c_str(), out_size);
            }
        } else {
            fprintf(stderr, "ERROR: cannot get output data (%d)\n", ret);
        }
    }

    // Cleanup
    ANeuroPilotTFLiteWrapper_free(tflite);
    ANeuralNetworksTFLiteOptions_free(options);

    if (times.empty()) {
        fprintf(stderr, "ERROR: No successful runs\n");
        return 1;
    }

    // Stats
    std::sort(times.begin(), times.end());
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double avg = sum / times.size();
    double min_v = times.front();
    double max_v = times.back();
    double median = times[times.size() / 2];

    printf("\nResults (n=%zu):\n", times.size());
    printf("  Avg:    %.2f ms\n", avg);
    printf("  Min:    %.2f ms\n", min_v);
    printf("  Max:    %.2f ms\n", max_v);
    printf("  Median: %.2f ms\n", median);
    printf("  33.3ms: %s\n", avg < 33.3 ? "YES" : "NO");

    return 0;
}
