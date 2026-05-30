#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <random>
#include <string>

enum class Metric {
    ANGULAR,
    EUCLIDEAN
};

struct SearchStats {
    int distance_computations = 0;
    int nodes_visited = 0;
};

class DiskANN {
public:
    DiskANN(const std::vector<std::vector<float>>& data, int R, int L, float alpha, Metric metric);

    int build();
    std::pair<std::vector<int>, SearchStats> search(const std::vector<float>& query, int k, int L_search, int start);

private:
    const std::vector<std::vector<float>>& data;
    int N, D, R, L;
    float alpha;
    Metric metric;
    std::vector<std::vector<int>> graph;

    float distance(int i, int j) const;
    float distance_query(const std::vector<float>& query, int j) const;
    std::vector<int> robust_prune(int p, const std::vector<int>& candidates, int R_param);
    std::pair<std::vector<std::pair<float, int>>, std::vector<int>> greedy_search(const std::vector<float>& query, int start, int L_param, SearchStats* stats = nullptr);
    int find_medoid();
    void build_pass(int medoid, float alpha_param);
};
