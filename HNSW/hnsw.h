#pragma once
#include <vector>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <unordered_set>

enum class Metric {
    ANGULAR,
    EUCLIDEAN
};

struct SearchStats {
    int distance_computations = 0;
    int nodes_visited = 0;
};

class HNSW {
public:
    HNSW(const std::vector<std::vector<float>>& data, int M, int ef_construction, Metric metric);

    void build();
    std::pair<std::vector<int>, SearchStats> search(const std::vector<float>& query, int k, int ef_search);

private:
    const std::vector<std::vector<float>>& data;
    int N, D, M, ef_construction;
    int M_max, M_max0;
    float ml;
    Metric metric;
    int entry_point;

    std::vector<std::vector<std::vector<int>>> graph;  // graph[layer][node] = neighbors
    std::vector<int> node_levels;

    float distance(int i, int j) const;
    float distance_query(const std::vector<float>& query, int j) const;
    int get_random_level();
    std::vector<std::pair<float, int>> search_layer(const std::vector<float>& query,
                                                      const std::vector<int>& entry_points,
                                                      int ef, int layer, SearchStats* stats = nullptr);
    void select_neighbors_heuristic(std::vector<std::pair<float, int>>& candidates,
                                     int M, int layer);
};
