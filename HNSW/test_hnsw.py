#!/usr/bin/env python3
import hnswlib
import numpy as np
import struct
import time
import sys

def load_fvecs(filename):
    """Load vectors from fvecs format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype=np.float32)

def load_ivecs(filename):
    """Load integer vectors from ivecs format"""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vec)
    return np.array(vectors, dtype=np.int32)

def calculate_recall(predictions, groundtruth, k):
    """Calculate recall@k"""
    total_recall = 0.0
    for i in range(len(predictions)):
        pred_set = set(predictions[i][:k])
        gt_set = set(groundtruth[i][:k])
        intersection = len(pred_set & gt_set)
        total_recall += intersection / k
    return total_recall / len(predictions)

def test_hnsw(data_dir, dataset_name, metric, M=64, ef_construction=200):
    """Test HNSW on a dataset"""
    print(f"Loading data for dataset: {dataset_name}")

    # Load data
    train = load_fvecs(f"{data_dir}/train_{dataset_name}.fvecs")
    test = load_fvecs(f"{data_dir}/test_{dataset_name}.fvecs")
    groundtruth = load_ivecs(f"{data_dir}/groundtruth_{dataset_name}.ivecs")

    print(f"Train: {train.shape}")
    print(f"Test: {test.shape}")
    print(f"Groundtruth: {groundtruth.shape}")

    # Set distance metric
    if metric == 'euclidean':
        space = 'l2'
    else:  # angular
        space = 'cosine'

    # Build HNSW index
    print(f"\nBuilding HNSW index with M={M}, ef_construction={ef_construction}, metric={space}")
    dim = train.shape[1]
    num_elements = train.shape[0]

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=num_elements, ef_construction=ef_construction, M=M)

    start_time = time.time()
    index.add_items(train, np.arange(num_elements))
    build_time = time.time() - start_time

    print(f"Build time: {build_time:.2f}s")

    return index, test, groundtruth

def run_search_tests(index, test, groundtruth, ef_search_list, k=100):
    """Run search tests with different ef_search values"""
    print(f"\n{'='*60}")
    print("Testing search performance")
    print('='*60)

    for ef_search in ef_search_list:
        print(f"\nTesting with ef_search={ef_search}")
        index.set_ef(ef_search)

        # Warm up
        for i in range(min(10, len(test))):
            index.knn_query(test[i], k=k)

        # Measure QPS
        start_time = time.time()
        labels, distances = index.knn_query(test, k=k)
        elapsed = time.time() - start_time

        qps = len(test) / elapsed
        recall = calculate_recall(labels, groundtruth, k)

        print(f"  QPS: {qps:.2f}, Recall@{k}: {recall:.6f}, Time: {elapsed:.2f}s")

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python test_hnsw.py <data_dir> <dataset_name> <metric>")
        print("  metric: angular or euclidean")
        print("Example: python test_hnsw.py /path/to/data glove angular")
        sys.exit(1)

    data_dir = sys.argv[1]
    dataset_name = sys.argv[2]
    metric = sys.argv[3]

    if metric not in ['angular', 'euclidean']:
        print(f"Invalid metric: {metric}")
        print("Must be 'angular' or 'euclidean'")
        sys.exit(1)

    # Build index
    index, test, groundtruth = test_hnsw(data_dir, dataset_name, metric, M=64, ef_construction=200)

    # Test with different ef_search values (same as DiskANN L_search)
    ef_search_list = [1,3,4,5,6,7,8, 9,10, 15, 20, 30, 50, 100, 150, 200, 300, 500]
    run_search_tests(index, test, groundtruth, ef_search_list, k=1)
