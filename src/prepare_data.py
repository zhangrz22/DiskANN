#!/usr/bin/env python3
import h5py
import numpy as np
import os

def save_fvecs(filename, data):
    """Highly optimized fvecs writer using numpy"""
    data = data.astype(np.float32)
    n, d = data.shape

    fvecs_data = np.empty((n, d + 1), dtype=np.float32)
    fvecs_data.view(np.int32)[:, 0] = d
    fvecs_data[:, 1:] = data

    with open(filename, 'wb') as f:
        fvecs_data.tofile(f)

def save_ivecs(filename, data):
    """Highly optimized ivecs writer using numpy"""
    data = data.astype(np.int32)
    n, d = data.shape

    ivecs_data = np.empty((n, d + 1), dtype=np.int32)
    ivecs_data[:, 0] = d
    ivecs_data[:, 1:] = data

    with open(filename, 'wb') as f:
        ivecs_data.tofile(f)

def prepare_dataset(hdf5_path, output_dir):
    """Convert HDF5 to binary format for C++"""
    # Extract dataset name from filename
    basename = os.path.basename(hdf5_path)
    dataset_name = basename.split('-')[0]  # e.g., 'glove', 'movielens10m', 'fashion'

    print(f"Loading {hdf5_path}...")
    with h5py.File(hdf5_path, 'r') as f:
        print(f"Keys in file: {list(f.keys())}")

        # Print all dataset shapes
        for key in f.keys():
            try:
                shape = f[key].shape if hasattr(f[key], 'shape') else f[key][()]
                print(f"  {key}: {shape}")
            except:
                pass

        train = np.array(f['train'], dtype=np.float32)
        test = np.array(f['test'], dtype=np.float32)
        neighbors = np.array(f['neighbors'], dtype=np.int32)

        # Check for size information (for datasets with flattened data)
        if 'size_train' in f and 'size_test' in f:
            size_train_data = f['size_train'][()]
            size_test_data = f['size_test'][()]

            # Check if this is a variable-length vector dataset
            if isinstance(size_train_data, np.ndarray) and len(size_train_data) > 1:
                print(f"Variable-length vector dataset detected")
                print(f"  Train vectors: {len(size_train_data)}, Test vectors: {len(size_test_data)}")

                # Extract variable-length vectors
                train_vecs = []
                offset = 0
                for size in size_train_data:
                    vec = train[offset:offset+size]
                    train_vecs.append(vec)
                    offset += size

                test_vecs = []
                offset = 0
                for size in size_test_data:
                    vec = test[offset:offset+size]
                    test_vecs.append(vec)
                    offset += size

                # Pad to max length
                max_dim = max(max(size_train_data), max(size_test_data))
                print(f"  Max dimension: {max_dim}, padding all vectors")

                train = np.array([np.pad(vec, (0, max_dim - len(vec))) for vec in train_vecs], dtype=np.float32)
                test = np.array([np.pad(vec, (0, max_dim - len(vec))) for vec in test_vecs], dtype=np.float32)

                print(f"  Padded train: {train.shape}, test: {test.shape}")
            else:
                # Handle scalar case (fixed-length vectors stored as 1D)
                size_train = int(size_train_data) if np.isscalar(size_train_data) else int(size_train_data[0])
                size_test = int(size_test_data) if np.isscalar(size_test_data) else int(size_test_data[0])
                print(f"Found size info: size_train={size_train}, size_test={size_test}")

            # Reshape if data is 1D
            if len(train.shape) == 1:
                dim = len(train) // size_train
                train = train.reshape(size_train, dim)
                print(f"Reshaped train to: {train.shape}")

            if len(test.shape) == 1:
                dim = len(test) // size_test
                test = test.reshape(size_test, dim)
                print(f"Reshaped test to: {test.shape}")

    print(f"Train shape: {train.shape}, dtype: {train.dtype}")
    print(f"Test shape: {test.shape}, dtype: {test.dtype}")
    print(f"Neighbors shape: {neighbors.shape}, dtype: {neighbors.dtype}")

    # Check if data is still 1D after processing
    if len(train.shape) == 1 or len(test.shape) == 1:
        print(f"Warning: Data is 1D and cannot be processed, skipping this dataset")
        return

    print(f"Train: {train.shape}, Test: {test.shape}, Neighbors: {neighbors.shape}")

    # Save with dataset-specific names
    print("Saving train vectors...")
    save_fvecs(f"{output_dir}/train_{dataset_name}.fvecs", train)

    print("Saving test vectors...")
    save_fvecs(f"{output_dir}/test_{dataset_name}.fvecs", test)

    print("Saving ground truth...")
    save_ivecs(f"{output_dir}/groundtruth_{dataset_name}.ivecs", neighbors)

    print(f"Done! Files saved with prefix '{dataset_name}'")

if __name__ == '__main__':
    datasets = [
        '/Users/zrz/Desktop/ANN/data/glove-25-angular.hdf5',
        '/Users/zrz/Desktop/ANN/data/coco-i2i-512-angular.hdf5',
        '/Users/zrz/Desktop/ANN/data/fashion-mnist-784-euclidean.hdf5'
    ]

    output_dir = '/Users/zrz/Desktop/ANN/data'

    for dataset in datasets:
        if os.path.exists(dataset):
            print(f"\n{'='*60}")
            prepare_dataset(dataset, output_dir)
        else:
            print(f"Warning: {dataset} not found, skipping...")

