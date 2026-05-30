#include "hnsw.h"
#include <iostream>
#include <cstring>

HNSW::HNSW(const std::vector<std::vector<float>>& data, int M, int ef_construction, Metric metric)
    : data(data), M(M), ef_construction(ef_construction), metric(metric) {
    N = data.size();
    D = data[0].size();
    M_max = M;
    M_max0 = M * 2;  // Layer 0 has more connections
    ml = 1.0 / log(2.0);  // Normalization factor for level assignment
    entry_point = -1;

    node_levels.resize(N);
    graph.resize(20);  // Support up to 20 layers
    for (int i = 0; i < 20; i++) {
        graph[i].resize(N);
    }
}

float HNSW::distance(int i, int j) const {
    const auto& vi = data[i];
    const auto& vj = data[j];

    if (metric == Metric::EUCLIDEAN) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            float diff = vi[k] - vj[k];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } else {  // ANGULAR
        float dot = 0.0f, norm_i = 0.0f, norm_j = 0.0f;
        for (int k = 0; k < D; k++) {
            dot += vi[k] * vj[k];
            norm_i += vi[k] * vi[k];
            norm_j += vj[k] * vj[k];
        }
        float cos_sim = dot / (std::sqrt(norm_i * norm_j) + 1e-10f);
        cos_sim = std::max(-1.0f, std::min(1.0f, cos_sim));
        return std::acos(cos_sim);
    }
}

float HNSW::distance_query(const std::vector<float>& query, int j) const {
    const auto& vj = data[j];

    if (metric == Metric::EUCLIDEAN) {
        float sum = 0.0f;
        for (int k = 0; k < D; k++) {
            float diff = query[k] - vj[k];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    } else {  // ANGULAR
        float dot = 0.0f, norm_q = 0.0f, norm_j = 0.0f;
        for (int k = 0; k < D; k++) {
            dot += query[k] * vj[k];
            norm_q += query[k] * query[k];
            norm_j += vj[k] * vj[k];
        }
        float cos_sim = dot / (std::sqrt(norm_q * norm_j) + 1e-10f);
        cos_sim = std::max(-1.0f, std::min(1.0f, cos_sim));
        return std::acos(cos_sim);
    }
}

int HNSW::get_random_level() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);

    double r = dis(gen);
    return (int)(-log(r) * ml);
}

std::vector<std::pair<float, int>> HNSW::search_layer(const std::vector<float>& query,
                                                        const std::vector<int>& entry_points,
                                                        int ef, int layer, SearchStats* stats) {
    std::unordered_set<int> visited;
    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> candidates(cmp);
    std::priority_queue<std::pair<float, int>> w;  // Max heap for results

    for (int ep : entry_points) {
        float d = distance_query(query, ep);
        if (stats) stats->distance_computations++;
        candidates.push({d, ep});
        w.push({d, ep});
        visited.insert(ep);
    }

    while (!candidates.empty()) {
        auto [curr_dist, curr] = candidates.top();
        candidates.pop();

        if (curr_dist > w.top().first) {
            break;
        }

        for (int neighbor : graph[layer][curr]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                float d = distance_query(query, neighbor);
                if (stats) stats->distance_computations++;

                if (d < w.top().first || w.size() < ef) {
                    candidates.push({d, neighbor});
                    w.push({d, neighbor});

                    if (w.size() > ef) {
                        w.pop();
                    }
                }
            }
        }
    }

    std::vector<std::pair<float, int>> result;
    while (!w.empty()) {
        result.push_back(w.top());
        w.pop();
    }
    std::reverse(result.begin(), result.end());

    if (stats) stats->nodes_visited += visited.size();

    return result;
}

void HNSW::select_neighbors_heuristic(std::vector<std::pair<float, int>>& candidates,
                                       int M, int layer) {
    // Simple heuristic: keep M closest neighbors
    std::sort(candidates.begin(), candidates.end());
    if (candidates.size() > M) {
        candidates.resize(M);
    }
}

void HNSW::build() {
    std::cout << "Building HNSW index with N=" << N << ", M=" << M
              << ", ef_construction=" << ef_construction << std::endl;

    for (int i = 0; i < N; i++) {
        if (i % 10000 == 0) {
            std::cout << "  Processing node " << i << "/" << N << std::endl;
        }

        int level = get_random_level();
        node_levels[i] = level;

        if (entry_point == -1) {
            entry_point = i;
            continue;
        }

        std::vector<int> entry_points = {entry_point};

        // Search from top to target layer
        for (int lc = std::min(level, node_levels[entry_point]); lc >= 0; lc--) {
            auto nearest = search_layer(data[i], entry_points, 1, lc);
            entry_points = {nearest[0].second};
        }

        // Insert at each layer from 0 to level
        for (int lc = 0; lc <= level; lc++) {
            auto candidates = search_layer(data[i], entry_points, ef_construction, lc);

            int M_layer = (lc == 0) ? M_max0 : M_max;
            select_neighbors_heuristic(candidates, M_layer, lc);

            // Add bidirectional links
            for (auto [d, neighbor] : candidates) {
                graph[lc][i].push_back(neighbor);
                graph[lc][neighbor].push_back(i);

                // Prune neighbor's connections if needed
                if (graph[lc][neighbor].size() > M_layer) {
                    std::vector<std::pair<float, int>> neighbor_candidates;
                    for (int n : graph[lc][neighbor]) {
                        neighbor_candidates.push_back({distance(neighbor, n), n});
                    }
                    select_neighbors_heuristic(neighbor_candidates, M_layer, lc);

                    graph[lc][neighbor].clear();
                    for (auto [d, n] : neighbor_candidates) {
                        graph[lc][neighbor].push_back(n);
                    }
                }
            }

            entry_points = {i};
        }

        if (level > node_levels[entry_point]) {
            entry_point = i;
        }
    }

    std::cout << "HNSW construction complete" << std::endl;
}

std::pair<std::vector<int>, SearchStats> HNSW::search(const std::vector<float>& query, int k, int ef_search) {
    SearchStats stats;
    std::vector<int> entry_points = {entry_point};

    // Search from top layer to layer 0
    for (int lc = node_levels[entry_point]; lc > 0; lc--) {
        auto nearest = search_layer(query, entry_points, 1, lc, &stats);
        entry_points = {nearest[0].second};
    }

    // Search at layer 0 with ef_search
    auto candidates = search_layer(query, entry_points, ef_search, 0, &stats);

    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)candidates.size()); i++) {
        result.push_back(candidates[i].second);
    }
    return {result, stats};
}
