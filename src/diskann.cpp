#include "diskann.h"
#include <iostream>
#include <cstring>
#include <numeric>

DiskANN::DiskANN(const std::vector<std::vector<float>>& data, int R, int L, float alpha, Metric metric)
    : data(data), R(R), L(L), alpha(alpha), metric(metric), entry(-1) {
    N = (int)data.size();
    D = N == 0 ? 0 : (int)data[0].size();
    graph.resize(N);
    reverse_graph.resize(N);
    active.assign(N, 1);
}

float DiskANN::distance(int i, int j) const {
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

float DiskANN::distance_query(const std::vector<float>& query, int j) const {
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

std::vector<int> DiskANN::robust_prune(int p, const std::vector<int>& candidates, int R_param) {
    std::unordered_set<int> cand_set(candidates.begin(), candidates.end());
    cand_set.erase(p);
    for (auto it = cand_set.begin(); it != cand_set.end();) {
        if (*it < 0 || *it >= N || !active[*it]) {
            it = cand_set.erase(it);
        } else {
            ++it;
        }
    }

    std::vector<int> result;
    while ((int)result.size() < R_param && !cand_set.empty()) {
        float min_dist = std::numeric_limits<float>::max();
        int best = -1;

        for (int c : cand_set) {
            float d = distance(p, c);
            if (d < min_dist) {
                min_dist = d;
                best = c;
            }
        }

        if (best == -1) break;

        result.push_back(best);
        cand_set.erase(best);

        std::vector<int> to_remove;
        for (int c : cand_set) {
            if (distance(p, c) > alpha * distance(best, c)) {
                to_remove.push_back(c);
            }
        }
        for (int c : to_remove) {
            cand_set.erase(c);
        }
    }

    return result;
}

std::pair<std::vector<std::pair<float, int>>, std::vector<int>> DiskANN::greedy_search(const std::vector<float>& query, int start, int L_param, SearchStats* stats) {
    start = normalize_start(start);
    if (start < 0) {
        return {{}, {}};
    }

    std::vector<char> visited_flag(N, 0);
    std::vector<int> visited_nodes;

    visited_flag[start] = true;
    visited_nodes.push_back(start);

    auto cmp = [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
        return a.first > b.first;
    };
    std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>, decltype(cmp)> candidates(cmp);

    float d = distance_query(query, start);
    if (stats) stats->distance_computations++;
    candidates.push({d, start});

    std::vector<std::pair<float, int>> result;
    result.push_back({d, start});

    while (!candidates.empty()) {
        auto [dist, current] = candidates.top();
        candidates.pop();

        // Early stopping
        if ((int)result.size() == L_param && dist > result.back().first) {
            break;
        }

        for (int neighbor : graph[current]) {
            if (neighbor >= 0 && neighbor < N && active[neighbor] && !visited_flag[neighbor]) {
                visited_flag[neighbor] = true;
                visited_nodes.push_back(neighbor);
                float d = distance_query(query, neighbor);
                if (stats) stats->distance_computations++;

                if ((int)result.size() < L_param || d < result.back().first) {
                    candidates.push({d, neighbor});

                    // Use binary search instead of sort
                    auto it = std::lower_bound(result.begin(), result.end(), std::make_pair(d, neighbor),
                        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                            return a.first < b.first;
                        });
                    result.insert(it, {d, neighbor});

                    if ((int)result.size() > L_param) {
                        result.resize(L_param);
                    }
                }
            }
        }
    }

    if (stats) stats->nodes_visited = visited_nodes.size();

    return {result, visited_nodes};
}

int DiskANN::find_medoid() {
    if (active_size() == 0) return -1;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, N - 1);

    int sample_size = std::min(1000, active_size());
    std::vector<int> samples;
    while ((int)samples.size() < sample_size) {
        int id = dis(gen);
        if (active[id]) samples.push_back(id);
    }

    float min_sum = std::numeric_limits<float>::max();
    int medoid = samples[0];
    for (int i : samples) {
        float sum = 0.0f;
        for (int j : samples) {
            if (i != j) sum += distance(i, j);
        }
        if (sum < min_sum) {
            min_sum = sum;
            medoid = i;
        }
    }
    return medoid;
}

void DiskANN::build_pass(int medoid, float alpha_param) {
    // Random permutation of nodes (critical!)
    std::vector<int> order(N);
    std::iota(order.begin(), order.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(order.begin(), order.end(), gen);

    for (int idx = 0; idx < N; idx++) {
        int i = order[idx];
        if (!active[i]) continue;
        if (idx % 10000 == 0) {
            std::cout << "  Processing node " << idx << "/" << N << std::endl;
        }

        auto [neighbors, visited_nodes] = greedy_search(data[i], medoid, L);

        // Use top-L neighbors as candidates (as per paper)
        std::vector<int> candidates;
        for (const auto& [dist, node] : neighbors) {
            candidates.push_back(node);
        }

        // Merge with existing neighbors (critical for Pass 2!)
        candidates.insert(candidates.end(), graph[i].begin(), graph[i].end());

        float old_alpha = alpha;
        alpha = alpha_param;
        set_neighbors(i, robust_prune(i, candidates, R));
        alpha = old_alpha;

        for (int j : graph[i]) {
            if (!active[j]) continue;
            if (std::find(graph[j].begin(), graph[j].end(), i) == graph[j].end()) {
                std::vector<int> updated = graph[j];
                updated.push_back(i);
                if ((int)updated.size() > R) {
                    old_alpha = alpha;
                    alpha = alpha_param;
                    updated = robust_prune(j, updated, R);
                    alpha = old_alpha;
                }
                set_neighbors(j, updated);
            }
        }
    }
}

int DiskANN::build() {
    std::cout << "Building Vamana graph with N=" << N << ", R=" << R << ", L=" << L << ", alpha=" << alpha << std::endl;
    if (N == 0) {
        entry = -1;
        return entry;
    }

    // Initialize random graph
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < N; i++) {
        if (!active[i]) continue;
        std::uniform_int_distribution<> dis(0, N - 1);
        std::unordered_set<int> neighbors;
        int target_degree = std::min(R, active_size() - 1);
        while ((int)neighbors.size() < target_degree) {
            int j = dis(gen);
            if (j != i && active[j]) neighbors.insert(j);
        }
        set_neighbors(i, std::vector<int>(neighbors.begin(), neighbors.end()));
    }

    int medoid = find_medoid();
    entry = medoid;
    std::cout << "Medoid: " << medoid << std::endl;

    std::cout << "\nPass 1: Building with alpha=1" << std::endl;
    build_pass(medoid, 1.0f);

    if (alpha > 1.0f) {
        std::cout << "\nPass 2: Building with alpha=" << alpha << std::endl;
        build_pass(medoid, alpha);
    }

    std::cout << "Graph construction complete" << std::endl;
    return medoid;
}

std::pair<std::vector<int>, SearchStats> DiskANN::search(const std::vector<float>& query, int k, int L_search, int start) {
    SearchStats stats;
    auto [neighbors, visited] = greedy_search(query, start, L_search, &stats);
    std::vector<int> result;
    for (int i = 0; i < std::min(k, (int)neighbors.size()); i++) {
        result.push_back(neighbors[i].second);
    }
    return {result, stats};
}

int DiskANN::insert(const std::vector<float>& vector, int start) {
    if (D == 0) {
        D = (int)vector.size();
    }
    if ((int)vector.size() != D) {
        std::cerr << "Insert dimension mismatch: expected " << D << ", got " << vector.size() << std::endl;
        return -1;
    }

    int id = N;
    data.push_back(vector);
    graph.emplace_back();
    reverse_graph.emplace_back();
    active.push_back(1);
    N++;

    if (active_size() == 1) {
        entry = id;
        return id;
    }

    int search_start = normalize_start(start);
    auto [nearest, visited] = greedy_search(vector, search_start, std::max(L, R + 1), nullptr);

    std::vector<int> candidates;
    for (const auto& item : nearest) candidates.push_back(item.second);
    candidates.insert(candidates.end(), visited.begin(), visited.end());
    set_neighbors(id, robust_prune(id, candidates, R));
    add_reverse_edges(id);

    if (entry < 0 || !active[entry]) {
        entry = find_medoid();
    }
    return id;
}

bool DiskANN::remove(int id) {
    if (id < 0 || id >= N || !active[id]) {
        return false;
    }

    std::vector<int> deleted_neighbors = graph[id];
    std::vector<int> affected;
    affected.reserve(reverse_graph[id].size());
    for (int node : reverse_graph[id]) {
        if (node >= 0 && node < N && active[node] && node != id) {
            affected.push_back(node);
        }
    }

    set_neighbors(id, {});
    active[id] = 0;
    reverse_graph[id].clear();

    for (int node : affected) {
        std::vector<int> candidates = graph[node];
        candidates.insert(candidates.end(), deleted_neighbors.begin(), deleted_neighbors.end());
        for (int neighbor : deleted_neighbors) {
            if (neighbor >= 0 && neighbor < N && active[neighbor]) {
                candidates.insert(candidates.end(), graph[neighbor].begin(), graph[neighbor].end());
            }
        }
        set_neighbors(node, robust_prune(node, candidates, R));
    }

    if (entry == id) {
        entry = find_medoid();
    }
    return true;
}

int DiskANN::active_size() const {
    return std::count(active.begin(), active.end(), 1);
}

int DiskANN::size() const {
    return N;
}

int DiskANN::entry_point() const {
    return entry;
}

bool DiskANN::contains_deleted_edges() const {
    for (int node = 0; node < N; node++) {
        if (!active[node]) continue;
        for (int neighbor : graph[node]) {
            if (neighbor < 0 || neighbor >= N || !active[neighbor]) {
                return true;
            }
        }
    }
    return false;
}

int DiskANN::normalize_start(int start) const {
    if (start >= 0 && start < N && active[start]) {
        return start;
    }
    if (entry >= 0 && entry < N && active[entry]) {
        return entry;
    }
    for (int i = 0; i < N; i++) {
        if (active[i]) return i;
    }
    return -1;
}

void DiskANN::add_reverse_edges(int node) {
    for (int neighbor : graph[node]) {
        if (neighbor < 0 || neighbor >= N || !active[neighbor]) continue;
        if (std::find(graph[neighbor].begin(), graph[neighbor].end(), node) == graph[neighbor].end()) {
            std::vector<int> updated = graph[neighbor];
            updated.push_back(node);
            set_neighbors(neighbor, updated);
        }
        if ((int)graph[neighbor].size() > R) {
            set_neighbors(neighbor, robust_prune(neighbor, graph[neighbor], R));
        }
    }
}

void DiskANN::set_neighbors(int node, const std::vector<int>& neighbors) {
    if (node < 0 || node >= N) return;

    for (int old_neighbor : graph[node]) {
        if (old_neighbor >= 0 && old_neighbor < (int)reverse_graph.size()) {
            reverse_graph[old_neighbor].erase(node);
        }
    }

    std::vector<int> filtered;
    filtered.reserve(neighbors.size());
    std::unordered_set<int> seen;
    for (int neighbor : neighbors) {
        if (neighbor >= 0 && neighbor < N && neighbor != node && active[neighbor] && seen.insert(neighbor).second) {
            filtered.push_back(neighbor);
            reverse_graph[neighbor].insert(node);
        }
    }
    graph[node] = std::move(filtered);
}
