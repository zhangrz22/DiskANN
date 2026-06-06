#include "diskann.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <unordered_set>
#include <algorithm>
#include <random>

struct Config {
    std::string data_dir;
    std::string dataset_name;
    std::string metric_str;
    bool dynamic_benchmark = false;
    int base_count = 10000;
    int op_count = 1000;
    int R = 64;
    int L = 100;
    float alpha = 1.2f;
};

float calculate_recall(const std::vector<std::vector<int>>& predictions,
                       const std::vector<std::vector<int>>& groundtruth, int k) {
    float total_recall = 0.0f;
    for (size_t i = 0; i < predictions.size(); i++) {
        std::unordered_set<int> pred_set(predictions[i].begin(), predictions[i].begin() + std::min(k, (int)predictions[i].size()));
        std::unordered_set<int> gt_set(groundtruth[i].begin(), groundtruth[i].begin() + k);

        int intersection = 0;
        for (int p : pred_set) {
            if (gt_set.count(p)) intersection++;
        }
        total_recall += (float)intersection / k;
    }
    return total_recall / predictions.size();
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <data_dir> <dataset_name> <metric> [options]\n";
    std::cerr << "  metric: angular or euclidean\n";
    std::cerr << "Options:\n";
    std::cerr << "  --dynamic              run insert/delete throughput benchmark\n";
    std::cerr << "  --base-count <n>       initial vectors for dynamic benchmark (default: 10000)\n";
    std::cerr << "  --op-count <n>         insert/delete operations to benchmark (default: 1000)\n";
    std::cerr << "  --R <n>                max graph degree (default: 64)\n";
    std::cerr << "  --L <n>                build/search candidate list size (default: 100)\n";
    std::cerr << "  --alpha <x>            Vamana prune alpha (default: 1.2)\n";
    std::cerr << "Example: " << program << " /home/xulin/data/ann glove angular --dynamic --base-count 10000 --op-count 1000\n";
}

bool parse_args(int argc, char** argv, Config& cfg) {
    if (argc < 4) return false;
    cfg.data_dir = argv[1];
    cfg.dataset_name = argv[2];
    cfg.metric_str = argv[3];

    for (int i = 4; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dynamic") {
            cfg.dynamic_benchmark = true;
        } else if (arg == "--base-count" && i + 1 < argc) {
            cfg.base_count = std::stoi(argv[++i]);
        } else if (arg == "--op-count" && i + 1 < argc) {
            cfg.op_count = std::stoi(argv[++i]);
        } else if (arg == "--R" && i + 1 < argc) {
            cfg.R = std::stoi(argv[++i]);
        } else if (arg == "--L" && i + 1 < argc) {
            cfg.L = std::stoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            cfg.alpha = std::stof(argv[++i]);
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << std::endl;
            return false;
        }
    }
    return true;
}

double seconds_since(std::chrono::high_resolution_clock::time_point start) {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int run_dynamic_benchmark(const std::vector<std::vector<float>>& train, Metric metric, const Config& cfg) {
    if ((int)train.size() < cfg.base_count + cfg.op_count) {
        std::cerr << "Not enough vectors for dynamic benchmark: need "
                  << cfg.base_count + cfg.op_count << ", have " << train.size() << std::endl;
        return 1;
    }

    std::vector<std::vector<float>> base(train.begin(), train.begin() + cfg.base_count);
    DiskANN index(base, cfg.R, cfg.L, cfg.alpha, metric);

    std::cout << "\n" << std::string(72, '=') << std::endl;
    std::cout << "Dynamic insert/delete benchmark" << std::endl;
    std::cout << std::string(72, '=') << std::endl;
    std::cout << "Dataset: " << cfg.dataset_name
              << ", base_count=" << cfg.base_count
              << ", op_count=" << cfg.op_count
              << ", R=" << cfg.R
              << ", L=" << cfg.L
              << ", alpha=" << cfg.alpha << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    int medoid = index.build();
    double build_time = seconds_since(start);
    std::cout << "Build time: " << build_time << "s" << std::endl;
    std::cout << "Initial entry point: " << medoid << std::endl;

    std::vector<int> inserted_ids;
    inserted_ids.reserve(cfg.op_count);
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < cfg.op_count; i++) {
        int id = index.insert(train[cfg.base_count + i], index.entry_point());
        if (id < 0) return 1;
        inserted_ids.push_back(id);
    }
    double insert_time = seconds_since(start);

    std::shuffle(inserted_ids.begin(), inserted_ids.end(), std::mt19937(42));
    int deleted = 0;
    start = std::chrono::high_resolution_clock::now();
    for (int id : inserted_ids) {
        if (index.remove(id)) deleted++;
    }
    double delete_time = seconds_since(start);

    std::cout << "Insert operations: " << cfg.op_count << std::endl;
    std::cout << "Insert time: " << insert_time << "s" << std::endl;
    std::cout << "Insert throughput: " << (cfg.op_count / insert_time) << " ops/s" << std::endl;
    std::cout << "Delete operations: " << deleted << std::endl;
    std::cout << "Delete time: " << delete_time << "s" << std::endl;
    std::cout << "Delete throughput: " << (deleted / delete_time) << " ops/s" << std::endl;
    std::cout << "Active vectors after benchmark: " << index.active_size() << std::endl;
    std::cout << "Deleted edge check: " << (index.contains_deleted_edges() ? "failed" : "passed") << std::endl;
    return 0;
}

int main(int argc, char** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) {
        print_usage(argv[0]);
        return 1;
    }

    Metric metric;
    if (cfg.metric_str == "euclidean") {
        metric = Metric::EUCLIDEAN;
    } else if (cfg.metric_str == "angular") {
        metric = Metric::ANGULAR;
    } else {
        std::cerr << "Invalid metric: " << cfg.metric_str << std::endl;
        std::cerr << "Must be 'angular' or 'euclidean'" << std::endl;
        return 1;
    }

    // Load data
    std::cout << "Loading data for dataset: " << cfg.dataset_name << std::endl;
    auto train = load_fvecs(cfg.data_dir + "/train_" + cfg.dataset_name + ".fvecs");

    if (cfg.dynamic_benchmark) {
        return run_dynamic_benchmark(train, metric, cfg);
    }

    auto test = load_fvecs(cfg.data_dir + "/test_" + cfg.dataset_name + ".fvecs");
    auto groundtruth = load_ivecs(cfg.data_dir + "/groundtruth_" + cfg.dataset_name + ".ivecs");

    std::cout << "Train: " << train.size() << " x " << train[0].size() << std::endl;
    std::cout << "Test: " << test.size() << " x " << test[0].size() << std::endl;
    std::cout << "Groundtruth: " << groundtruth.size() << " x " << groundtruth[0].size() << std::endl;

    // Build index
    std::cout << "\nBuilding DiskANN index..." << std::endl;
    DiskANN index(train, cfg.R, cfg.L, cfg.alpha, metric);
    auto start = std::chrono::high_resolution_clock::now();
    int medoid = index.build();
    auto end = std::chrono::high_resolution_clock::now();
    double build_time = std::chrono::duration<double>(end - start).count();
    std::cout << "Build time: " << build_time << "s" << std::endl;

    // Test with different L_search values
    std::vector<int> L_search_list = {1, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50, 100, 150, 200, 300, 500};
    int k = 1;

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "Testing search performance" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    for (int L_search : L_search_list) {
        std::cout << "\nTesting with L_search=" << L_search << std::endl;

        // Warm up
        for (int i = 0; i < std::min(10, (int)test.size()); i++) {
            index.search(test[i], k, L_search, medoid);
        }

        // Measure QPS
        start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int>> predictions;
        long long total_dist_comps = 0;
        long long total_nodes_visited = 0;
        for (const auto& query : test) {
            auto [result, stats] = index.search(query, k, L_search, medoid);
            predictions.push_back(result);
            total_dist_comps += stats.distance_computations;
            total_nodes_visited += stats.nodes_visited;
        }
        end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();

        double qps = test.size() / elapsed;
        float recall = calculate_recall(predictions, groundtruth, k);
        double avg_dist_comps = (double)total_dist_comps / test.size();
        double avg_nodes_visited = (double)total_nodes_visited / test.size();

        std::cout << "  QPS: " << qps << ", Recall@" << k << ": " << recall
                  << ", Time: " << elapsed << "s" << std::endl;
        std::cout << "  Avg distance computations: " << avg_dist_comps
                  << ", Avg nodes visited: " << avg_nodes_visited << std::endl;
    }

    return 0;
}
