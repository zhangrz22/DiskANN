#include "diskann.h"
#include <iostream>
#include <unordered_set>
#include <cstring>

DiskANN::DiskANN(const std::vector<std::vector<float>>& data, int R, int L, float alpha, Metric metric)
    : data(data), R(R), L(L), alpha(alpha), metric(metric) {
    N = data.size();
    D = data[0].size();
    graph.resize(N);
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

    std::vector<int> result;
    while (result.size() < R_param && !cand_set.empty()) {
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
    static thread_local std::vector<bool> visited_flag(N, false);
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
        if (result.size() == L_param && dist > result.back().first) {
            break;
        }

        for (int neighbor : graph[current]) {
            if (!visited_flag[neighbor]) {
                visited_flag[neighbor] = true;
                visited_nodes.push_back(neighbor);
                float d = distance_query(query, neighbor);
                if (stats) stats->distance_computations++;

                if (result.size() < L_param || d < result.back().first) {
                    candidates.push({d, neighbor});

                    // Use binary search instead of sort
                    auto it = std::lower_bound(result.begin(), result.end(), std::make_pair(d, neighbor),
                        [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                            return a.first < b.first;
                        });
                    result.insert(it, {d, neighbor});

                    if (result.size() > L_param) {
                        result.resize(L_param);
                    }
                }
            }
        }
    }

    // Clear visited flags
    for (int node : visited_nodes) {
        visited_flag[node] = false;
    }

    if (stats) stats->nodes_visited = visited_nodes.size();

    return {result, visited_nodes};
}

int DiskANN::find_medoid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, N - 1);

    int sample_size = std::min(1000, N);
    std::vector<int> samples;
    for (int i = 0; i < sample_size; i++) {
        samples.push_back(dis(gen));
    }

    float min_sum = std::numeric_limits<float>::max();
    int medoid = 0;
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
        graph[i] = robust_prune(i, candidates, R);
        alpha = old_alpha;

        for (int j : graph[i]) {
            if (std::find(graph[j].begin(), graph[j].end(), i) == graph[j].end()) {
                graph[j].push_back(i);
                if (graph[j].size() > R) {
                    std::vector<int> rev_candidates = graph[j];
                    old_alpha = alpha;
                    alpha = alpha_param;
                    graph[j] = robust_prune(j, rev_candidates, R);
                    alpha = old_alpha;
                }
            }
        }
    }
}

int DiskANN::build() {
    std::cout << "Building Vamana graph with N=" << N << ", R=" << R << ", L=" << L << ", alpha=" << alpha << std::endl;

    // Initialize random graph
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < N; i++) {
        std::uniform_int_distribution<> dis(0, N - 1);
        std::unordered_set<int> neighbors;
        while (neighbors.size() < std::min(R, N - 1)) {
            int j = dis(gen);
            if (j != i) neighbors.insert(j);
        }
        graph[i] = std::vector<int>(neighbors.begin(), neighbors.end());
    }

    int medoid = find_medoid();
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
