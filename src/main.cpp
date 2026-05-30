#include "diskann.h"
#include "utils.h"
#include <iostream>
#include <chrono>
#include <unordered_set>

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

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_dir> <dataset_name> <metric>" << std::endl;
        std::cerr << "  metric: angular or euclidean" << std::endl;
        std::cerr << "Example: " << argv[0] << " /path/to/data glove angular" << std::endl;
        return 1;
    }

    std::string data_dir = argv[1];
    std::string dataset_name = argv[2];
    std::string metric_str = argv[3];

    Metric metric;
    if (metric_str == "euclidean") {
        metric = Metric::EUCLIDEAN;
    } else if (metric_str == "angular") {
        metric = Metric::ANGULAR;
    } else {
        std::cerr << "Invalid metric: " << metric_str << std::endl;
        std::cerr << "Must be 'angular' or 'euclidean'" << std::endl;
        return 1;
    }

    // Load data
    std::cout << "Loading data for dataset: " << dataset_name << std::endl;
    auto train = load_fvecs(data_dir + "/train_" + dataset_name + ".fvecs");
    auto test = load_fvecs(data_dir + "/test_" + dataset_name + ".fvecs");
    auto groundtruth = load_ivecs(data_dir + "/groundtruth_" + dataset_name + ".ivecs");

    std::cout << "Train: " << train.size() << " x " << train[0].size() << std::endl;
    std::cout << "Test: " << test.size() << " x " << test[0].size() << std::endl;
    std::cout << "Groundtruth: " << groundtruth.size() << " x " << groundtruth[0].size() << std::endl;

    // Build index
    std::cout << "\nBuilding DiskANN index..." << std::endl;
    DiskANN index(train, 64, 100, 1.2, metric);
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
