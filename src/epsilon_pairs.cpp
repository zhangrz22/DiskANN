#include "utils.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

enum class Metric {
    ANGULAR,
    EUCLIDEAN
};

struct NormData {
    std::vector<float> squared_norms;
};

struct PairSearchStats {
    uint64_t pairs_found = 0;
    uint64_t distance_computations = 0;
    double elapsed_seconds = 0.0;
};

Metric parse_metric(const std::string& metric) {
    if (metric == "euclidean") {
        return Metric::EUCLIDEAN;
    }
    if (metric == "angular") {
        return Metric::ANGULAR;
    }
    std::cerr << "Invalid metric: " << metric << std::endl;
    std::cerr << "Must be 'angular' or 'euclidean'" << std::endl;
    std::exit(1);
}

NormData precompute_norms(const std::vector<std::vector<float>>& data) {
    NormData norms;
    norms.squared_norms.resize(data.size(), 0.0f);
    for (size_t i = 0; i < data.size(); i++) {
        float sum = 0.0f;
        for (float value : data[i]) {
            sum += value * value;
        }
        norms.squared_norms[i] = sum;
    }
    return norms;
}

bool euclidean_within_epsilon(const std::vector<float>& a,
                              const std::vector<float>& b,
                              float epsilon_squared) {
    float sum = 0.0f;
    for (size_t k = 0; k < a.size(); k++) {
        float diff = a[k] - b[k];
        sum += diff * diff;
        if (sum >= epsilon_squared) {
            return false;
        }
    }
    return true;
}

bool angular_within_epsilon(const std::vector<float>& a,
                            const std::vector<float>& b,
                            float norm_a_squared,
                            float norm_b_squared,
                            float cosine_threshold) {
    if (norm_a_squared <= 0.0f || norm_b_squared <= 0.0f) {
        return false;
    }

    float dot = 0.0f;
    for (size_t k = 0; k < a.size(); k++) {
        dot += a[k] * b[k];
    }

    float cosine = dot / (std::sqrt(norm_a_squared * norm_b_squared) + 1e-10f);
    return cosine > cosine_threshold;
}

PairSearchStats find_epsilon_pairs(const std::vector<std::vector<float>>& data,
                                   Metric metric,
                                   float epsilon,
                                   size_t block_size,
                                   const NormData& norms,
                                   std::ofstream* pair_output = nullptr) {
    const size_t n = data.size();
    PairSearchStats stats;
    const float epsilon_squared = epsilon * epsilon;
    const float cosine_threshold = std::cos(epsilon);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t block_i = 0; block_i < n; block_i += block_size) {
        const size_t end_i = std::min(block_i + block_size, n);
        for (size_t block_j = block_i; block_j < n; block_j += block_size) {
            const size_t end_j = std::min(block_j + block_size, n);

            for (size_t i = block_i; i < end_i; i++) {
                const size_t j_begin = (block_i == block_j) ? i + 1 : block_j;
                for (size_t j = j_begin; j < end_j; j++) {
                    stats.distance_computations++;
                    bool within_epsilon = false;
                    if (metric == Metric::EUCLIDEAN) {
                        within_epsilon = euclidean_within_epsilon(data[i], data[j], epsilon_squared);
                    } else {
                        within_epsilon = angular_within_epsilon(data[i], data[j],
                                                                norms.squared_norms[i],
                                                                norms.squared_norms[j],
                                                                cosine_threshold);
                    }

                    if (within_epsilon) {
                        stats.pairs_found++;
                        if (pair_output) {
                            *pair_output << epsilon << ' ' << i << ' ' << j << '\n';
                        }
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.elapsed_seconds = std::chrono::duration<double>(end - start).count();
    std::cout << std::endl;
    return stats;
}

std::vector<float> get_epsilon_list(const std::string& dataset_name, Metric metric) {
    if (metric == Metric::EUCLIDEAN) {
        if (dataset_name == "fashion") {
            return {800.0f, 1000.0f, 1200.0f, 1400.0f, 1600.0f};
        }
        return {0.5f, 1.0f, 2.0f, 4.0f, 8.0f};
    }

    if (dataset_name == "coco") {
        return {0.12f, 0.18f, 0.24f, 0.30f, 0.36f};
    }
    return {0.12f, 0.16f, 0.20f, 0.24f, 0.28f};
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <dataset_name> <metric> [--epsilon value] [--sample-size N] [--block-size N] [--output pairs.txt]" << std::endl;
        std::cerr << "  metric: angular or euclidean" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/data fashion euclidean --epsilon 1200 --sample-size 10000 --block-size 2048" << std::endl;
        std::cerr << "Legacy positional form is also supported: " << argv[0] << " /path/to/data fashion euclidean 10000 2048" << std::endl;
        return 1;
    }

    const std::string data_dir = argv[1];
    const std::string dataset_name = argv[2];
    const Metric metric = parse_metric(argv[3]);
    std::optional<size_t> sample_size;
    size_t block_size = 2048;
    std::optional<float> single_epsilon;
    std::string output_path;

    int argi = 4;
    if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
        sample_size = std::stoull(argv[argi++]);
        if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
            block_size = std::stoull(argv[argi++]);
        }
        if (argi < argc && std::string(argv[argi]).rfind("--", 0) != 0) {
            output_path = argv[argi++];
        }
    }

    while (argi < argc) {
        std::string arg = argv[argi++];
        if (arg == "--epsilon") {
            if (argi >= argc) {
                std::cerr << "--epsilon requires a value" << std::endl;
                return 1;
            }
            single_epsilon = std::stof(argv[argi++]);
        } else if (arg == "--sample-size") {
            if (argi >= argc) {
                std::cerr << "--sample-size requires a value" << std::endl;
                return 1;
            }
            sample_size = std::stoull(argv[argi++]);
        } else if (arg == "--block-size") {
            if (argi >= argc) {
                std::cerr << "--block-size requires a value" << std::endl;
                return 1;
            }
            block_size = std::stoull(argv[argi++]);
        } else if (arg == "--output") {
            if (argi >= argc) {
                std::cerr << "--output requires a path" << std::endl;
                return 1;
            }
            output_path = argv[argi++];
        } else {
            std::cerr << "Unexpected argument: " << arg << std::endl;
            return 1;
        }
    }

    if (sample_size && *sample_size == 0) {
        std::cerr << "sample_size must be positive" << std::endl;
        return 1;
    }
    if (block_size == 0) {
        std::cerr << "block_size must be positive" << std::endl;
        return 1;
    }
    if (single_epsilon && (*single_epsilon <= 0.0f || !std::isfinite(*single_epsilon))) {
        std::cerr << "epsilon must be a positive finite number" << std::endl;
        return 1;
    }

    std::cout << "Loading data for dataset: " << dataset_name << std::endl;
    const std::string fvecs_path = data_dir + "/train_" + dataset_name + ".fvecs";
    auto data = load_fvecs(fvecs_path);
    if (data.empty()) {
        std::cerr << "No vectors found in " << fvecs_path << std::endl;
        return 1;
    }
    const size_t full_size = data.size();
    if (sample_size && data.size() > *sample_size) {
        data.resize(*sample_size);
    }

    std::cout << "Train: " << full_size << " x " << data[0].size() << std::endl;
    if (sample_size) {
        std::cout << "Sample: " << data.size() << " x " << data[0].size() << std::endl;
    } else {
        std::cout << "Search set: full training set" << std::endl;
    }
    std::cout << "Block size: " << block_size << std::endl;

    const NormData norms = (metric == Metric::ANGULAR) ? precompute_norms(data) : NormData{};
    const std::vector<float> epsilon_list = single_epsilon ? std::vector<float>{*single_epsilon}
                                                           : get_epsilon_list(dataset_name, metric);
    std::ofstream pair_output;
    if (!output_path.empty()) {
        pair_output.open(output_path);
        if (!pair_output) {
            std::cerr << "Cannot open output file: " << output_path << std::endl;
            return 1;
        }
    }

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing epsilon-pair search performance" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    for (float epsilon : epsilon_list) {
        std::cout << "\nTesting with epsilon=" << epsilon << std::endl;

        auto stats = find_epsilon_pairs(data, metric, epsilon, block_size, norms,
                                        pair_output.is_open() ? &pair_output : nullptr);
        const double throughput = stats.distance_computations / std::max(stats.elapsed_seconds, 1e-9);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Pair checks/s: " << throughput
                  << ", Pairs found: " << stats.pairs_found
                  << ", Time: " << stats.elapsed_seconds << "s" << std::endl;
        std::cout << "  Distance computations: " << stats.distance_computations << std::endl;
    }

    if (pair_output.is_open()) {
        std::cout << "Pairs written to: " << output_path << std::endl;
    }

    return 0;
}
