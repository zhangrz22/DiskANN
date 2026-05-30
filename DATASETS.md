# 数据集文档

本文档详细介绍了DiskANN实验中使用的三个基准数据集的来源、结构和转换格式。

## 1. 数据集概览

| 数据集 | 训练集大小 | 维度 | 距离度量 | 数据类型 |
|--------|-----------|------|----------|----------|
| GloVe-25 | 1,183,514 | 25 | Angular | 文本词向量 |
| COCO-512 | 113,287 | 512 | Angular | 图像特征 |
| Fashion-MNIST | 60,000 | 784 | Euclidean | 图像像素 |

---

## 2. GloVe-25-Angular

### 2.1 数据集简介

**GloVe (Global Vectors for Word Representation)** 是斯坦福大学开发的无监督词向量学习算法。该数据集包含预训练的25维词向量，通过在大规模语料库上训练得到。

- **来源**: Stanford NLP Group
- **用途**: 自然语言处理任务（词语相似度、文本分类等）
- **特点**: 低维度（25维），大规模（118万向量）

### 2.2 原始HDF5结构

```
Keys: ['distances', 'neighbors', 'test', 'train']

数据形状:
- train: (1183514, 25)      # 118万训练向量
- test: (10000, 25)          # 1万查询向量
- neighbors: (10000, 100)    # 每个查询的100个真实最近邻索引
- distances: (10000, 100)    # 对应的距离值
```

### 2.3 距离度量

使用**Angular距离**（余弦相似度的角度形式）:

```
angular_distance = arccos(cosine_similarity)
              = arccos(dot(u,v) / (||u|| * ||v||))
```

---

## 3. COCO-I2I-512-Angular

### 3.1 数据集简介

**COCO (Common Objects in Context)** 是大规模物体检测、分割和图像描述数据集。该变体使用深度神经网络提取的512维图像特征向量，用于图像到图像（Image-to-Image）的相似度检索。

- **来源**: Microsoft COCO Dataset
- **用途**: 图像相似度检索、视觉搜索
- **特点**: 中等维度（512维），中等规模（11.3万向量）

### 3.2 原始HDF5结构

```
Keys: ['distances', 'neighbors', 'test', 'train']

数据形状:
- train: (113287, 512)       # 11.3万训练向量
- test: (10000, 512)         # 1万查询向量
- neighbors: (10000, 100)    # 每个查询的100个真实最近邻索引
- distances: (10000, 100)    # 对应的距离值
```

### 3.3 距离度量

同样使用**Angular距离**，适合深度学习特征向量的相似度度量。

---

## 4. Fashion-MNIST-784-Euclidean

### 4.1 数据集简介

**Fashion-MNIST** 是MNIST的现代替代品，包含10类时尚物品的28×28灰度图像。每个图像展平为784维向量（28×28=784）。

- **来源**: Zalando Research
- **用途**: 图像分类、相似度检索基准测试
- **特点**: 高维度（784维），中等规模（6万向量）

### 4.2 原始HDF5结构

```
Keys: ['distances', 'neighbors', 'test', 'train']

数据形状:
- train: (60000, 784)        # 6万训练向量
- test: (10000, 784)         # 1万查询向量
- neighbors: (10000, 100)    # 每个查询的100个真实最近邻索引
- distances: (10000, 100)    # 对应的距离值
```

### 4.3 距离度量

使用**欧式距离（L2距离）**:

```
euclidean_distance = sqrt(sum((u[i] - v[i])^2))
```

这是图像像素空间中最常用的距离度量。

---

## 5. 转换后的二进制格式

所有三个数据集经过`prepare_data.py`处理后，都转换为相同的二进制格式，便于C++高效读取。

### 5.1 .fvecs格式（浮点向量）

用于存储`train`和`test`数据。

**文件结构**（每个向量）:
```
[4字节 int32: 维度d] [d个 float32: 向量数据]
```

**示例**（25维向量）:
```
Bytes 0-3:   0x00000019  (整数25，表示维度)
Bytes 4-7:   0x3f800000  (float32: 第1个分量)
Bytes 8-11:  0x40000000  (float32: 第2个分量)
...
Bytes 100-103: 0x3f000000 (float32: 第25个分量)
```

### 5.2 .ivecs格式（整数向量）

用于存储`groundtruth`（真实最近邻索引）。

**文件结构**（每个向量）:
```
[4字节 int32: 维度d] [d个 int32: 整数数据]
```

**示例**（100个最近邻索引）:
```
Bytes 0-3:   0x00000064  (整数100，表示维度)
Bytes 4-7:   0x00001234  (int32: 第1个邻居索引)
Bytes 8-11:  0x00005678  (int32: 第2个邻居索引)
...
```

### 5.3 转换后的文件列表

每个数据集生成3个文件:

```
/Users/zrz/Desktop/ANN/data/
├── train_glove.fvecs          # GloVe训练集
├── test_glove.fvecs           # GloVe查询集
├── groundtruth_glove.ivecs    # GloVe真实最近邻
├── train_coco.fvecs           # COCO训练集
├── test_coco.fvecs            # COCO查询集
├── groundtruth_coco.ivecs     # COCO真实最近邻
├── train_fashion.fvecs        # Fashion训练集
├── test_fashion.fvecs         # Fashion查询集
└── groundtruth_fashion.ivecs  # Fashion真实最近邻
```

---

## 6. 数据集选择的合理性

这三个数据集的组合非常适合评估DiskANN算法:

1. **维度多样性**: 25维（低）、512维（中）、784维（高）
2. **规模多样性**: 6万、11.3万、118万向量
3. **距离度量多样性**: Angular（2个）、Euclidean（1个）
4. **应用场景多样性**: 文本、图像特征、原始像素

这种多样性确保了实验结果的全面性和算法的鲁棒性。
