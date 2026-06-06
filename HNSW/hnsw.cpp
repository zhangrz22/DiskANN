#include "hnsw.h"
#include <iostream>
#include <cstring>

HNSW::HNSW(const std::vector<std::vector<float>>& data, int M, int ef_construction, Metric metric)
    : data(data), M(M), ef_construction(ef_construction), metric(metric) {
    N = (int)data.size();
    D = N == 0 ? 0 : (int)data[0].size();
    M_max = M;
    M_max0 = M * 2;  // Layer 0 has more connections
    ml = 1.0 / log(2.0);  // Normalization factor for level assignment
    entry_point = -1;

    node_levels.resize(N);
    active.assign(N, 1);
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
        if (ep < 0 || ep >= N || !active[ep]) continue;
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
            if (neighbor >= 0 && neighbor < N && active[neighbor] && visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                float d = distance_query(query, neighbor);
                if (stats) stats->distance_computations++;

                if (d < w.top().first || (int)w.size() < ef) {
                    candidates.push({d, neighbor});
                    w.push({d, neighbor});

                    if ((int)w.size() > ef) {
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
    if ((int)candidates.size() > M) {
        candidates.resize(M);
    }
}

void HNSW::build() {
    std::cout << "Building HNSW index with N=" << N << ", M=" << M
              << ", ef_construction=" << ef_construction << std::endl;

    std::vector<std::vector<float>> original_data = data;
    data.clear();
    N = 0;
    D = original_data.empty() ? 0 : (int)original_data[0].size();
    entry_point = -1;
    node_levels.clear();
    active.clear();
    for (auto& layer : graph) {
        layer.clear();
    }

    for (int i = 0; i < (int)original_data.size(); i++) {
        if (i % 10000 == 0) {
            std::cout << "  Processing node " << i << "/" << original_data.size() << std::endl;
        }
        insert(original_data[i]);
    }

    std::cout << "HNSW construction complete" << std::endl;
}

std::pair<std::vector<int>, SearchStats> HNSW::search(const std::vector<float>& query, int k, int ef_search) {
    SearchStats stats;
    int ep = get_valid_entry_point();
    if (ep < 0) return {{}, stats};

    std::vector<int> entry_points = {ep};

    // Search from top layer to layer 0
    for (int lc = node_levels[ep]; lc > 0; lc--) {
        auto nearest = search_layer(query, entry_points, 1, lc, &stats);
        if (nearest.empty()) continue;
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

int HNSW::insert(const std::vector<float>& vector) {
    if (D == 0) {
        D = (int)vector.size();
    }
    if ((int)vector.size() != D) {
        std::cerr << "Insert dimension mismatch: expected " << D << ", got " << vector.size() << std::endl;
        return -1;
    }

    int id = N;
    data.push_back(vector);
    node_levels.push_back(0);
    active.push_back(1);
    for (auto& layer : graph) {
        layer.emplace_back();
    }
    N++;

    int level = std::min(get_random_level(), (int)graph.size() - 1);
    node_levels[id] = level;

    int ep = get_valid_entry_point();
    if (ep == -1) {
        entry_point = id;
        return id;
    }

    std::vector<int> entry_points = {ep};

    for (int lc = std::min(level, node_levels[ep]); lc >= 0; lc--) {
        auto nearest = search_layer(vector, entry_points, 1, lc);
        if (!nearest.empty()) {
            entry_points = {nearest[0].second};
        }
    }

    for (int lc = 0; lc <= level; lc++) {
        auto candidates = search_layer(vector, entry_points, ef_construction, lc);

        int M_layer = (lc == 0) ? M_max0 : M_max;
        select_neighbors_heuristic(candidates, M_layer, lc);

        for (auto [d, neighbor] : candidates) {
            graph[lc][id].push_back(neighbor);
            graph[lc][neighbor].push_back(id);

            if ((int)graph[lc][neighbor].size() > M_layer) {
                std::vector<std::pair<float, int>> neighbor_candidates;
                for (int n : graph[lc][neighbor]) {
                    if (n >= 0 && n < N && active[n] && n != neighbor) {
                        neighbor_candidates.push_back({distance(neighbor, n), n});
                    }
                }
                select_neighbors_heuristic(neighbor_candidates, M_layer, lc);

                graph[lc][neighbor].clear();
                for (auto [dist, n] : neighbor_candidates) {
                    graph[lc][neighbor].push_back(n);
                }
            }
        }

        entry_points = {id};
    }

    if (level > node_levels[ep]) {
        entry_point = id;
    }
    return id;
}

bool HNSW::remove(int id) {
    if (id < 0 || id >= N || !active[id]) {
        return false;
    }

    std::vector<std::vector<int>> deleted_neighbors_by_layer(graph.size());
    for (int layer = 0; layer < (int)graph.size(); layer++) {
        if (id < (int)graph[layer].size()) {
            deleted_neighbors_by_layer[layer] = graph[layer][id];
        }
    }

    active[id] = 0;

    for (int layer = 0; layer < (int)graph.size(); layer++) {
        if (id < (int)graph[layer].size()) {
            graph[layer][id].clear();
        }
        repair_layer_after_delete(id, layer, deleted_neighbors_by_layer[layer]);
    }

    for (int layer = 0; layer < (int)graph.size(); layer++) {
        for (int node = 0; node < N; node++) {
            if (active[node]) {
                auto& adj = graph[layer][node];
                adj.erase(std::remove(adj.begin(), adj.end(), id), adj.end());
            }
        }
    }

    if (entry_point == id) {
        entry_point = get_valid_entry_point();
    }
    return true;
}

int HNSW::active_size() const {
    return std::count(active.begin(), active.end(), 1);
}

int HNSW::size() const {
    return N;
}

bool HNSW::contains_deleted_edges() const {
    for (int layer = 0; layer < (int)graph.size(); layer++) {
        for (int node = 0; node < N; node++) {
            if (!active[node]) continue;
            for (int neighbor : graph[layer][node]) {
                if (neighbor < 0 || neighbor >= N || !active[neighbor]) {
                    return true;
                }
            }
        }
    }
    return false;
}

int HNSW::get_valid_entry_point() const {
    if (entry_point >= 0 && entry_point < N && active[entry_point]) {
        return entry_point;
    }
    int best = -1;
    int best_level = -1;
    for (int i = 0; i < N; i++) {
        if (active[i] && node_levels[i] > best_level) {
            best = i;
            best_level = node_levels[i];
        }
    }
    return best;
}

void HNSW::repair_layer_after_delete(int deleted_id, int layer, const std::vector<int>& deleted_neighbors) {
    std::vector<int> affected;
    for (int neighbor : deleted_neighbors) {
        add_unique(affected, neighbor);
    }
    for (int node = 0; node < N; node++) {
        if (!active[node] || node == deleted_id) continue;
        if (std::find(graph[layer][node].begin(), graph[layer][node].end(), deleted_id) != graph[layer][node].end()) {
            add_unique(affected, node);
        }
    }

    int M_layer = (layer == 0) ? M_max0 : M_max;
    for (int node : affected) {
        if (node < 0 || node >= N || !active[node]) continue;

        std::vector<int> candidate_ids;
        for (int current : graph[layer][node]) {
            if (current != deleted_id) add_unique(candidate_ids, current);
        }
        for (int neighbor : deleted_neighbors) {
            if (neighbor != node && neighbor != deleted_id && neighbor >= 0 && neighbor < N && active[neighbor]) {
                add_unique(candidate_ids, neighbor);
                for (int second_hop : graph[layer][neighbor]) {
                    if (second_hop != node && second_hop != deleted_id) {
                        add_unique(candidate_ids, second_hop);
                    }
                }
            }
        }

        std::vector<std::pair<float, int>> candidates;
        candidates.reserve(candidate_ids.size());
        for (int candidate : candidate_ids) {
            if (candidate >= 0 && candidate < N && active[candidate] && candidate != node) {
                candidates.push_back({distance(node, candidate), candidate});
            }
        }

        select_neighbors_heuristic(candidates, M_layer, layer);
        graph[layer][node].clear();
        for (auto [dist, candidate] : candidates) {
            graph[layer][node].push_back(candidate);
        }
    }

    for (int node : affected) {
        if (node < 0 || node >= N || !active[node]) continue;
        std::vector<int> neighbors = graph[layer][node];
        for (int neighbor : neighbors) {
            if (neighbor < 0 || neighbor >= N || !active[neighbor]) continue;
            if (std::find(graph[layer][neighbor].begin(), graph[layer][neighbor].end(), node) == graph[layer][neighbor].end()) {
                graph[layer][neighbor].push_back(node);
                prune_node_neighbors(neighbor, layer);
            }
        }
        prune_node_neighbors(node, layer);
    }
}

void HNSW::prune_node_neighbors(int node, int layer) {
    if (node < 0 || node >= N || !active[node]) return;
    int M_layer = (layer == 0) ? M_max0 : M_max;

    std::vector<std::pair<float, int>> candidates;
    for (int neighbor : graph[layer][node]) {
        if (neighbor >= 0 && neighbor < N && active[neighbor] && neighbor != node) {
            candidates.push_back({distance(node, neighbor), neighbor});
        }
    }
    select_neighbors_heuristic(candidates, M_layer, layer);

    graph[layer][node].clear();
    std::unordered_set<int> seen;
    for (auto [dist, neighbor] : candidates) {
        if (seen.insert(neighbor).second) {
            graph[layer][node].push_back(neighbor);
        }
    }
}

void HNSW::add_unique(std::vector<int>& values, int value) const {
    if (value >= 0 && value < N && active[value] &&
        std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}
