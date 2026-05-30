#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

inline std::vector<std::vector<float>> load_fvecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<float>> data;
    while (in.peek() != EOF) {
        int dim;
        in.read((char*)&dim, sizeof(int));
        std::vector<float> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(float));
        data.push_back(vec);
    }
    return data;
}

inline std::vector<std::vector<int>> load_ivecs(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Cannot open " << filename << std::endl;
        exit(1);
    }

    std::vector<std::vector<int>> data;
    while (in.peek() != EOF) {
        int dim;
        in.read((char*)&dim, sizeof(int));
        std::vector<int> vec(dim);
        in.read((char*)vec.data(), dim * sizeof(int));
        data.push_back(vec);
    }
    return data;
}
