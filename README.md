# LoRS_Distill

本 README 是基于当前仓库代码的“真实实现状态说明文档”。它的目标不是描述理想算法设计，而是尽量准确回答：

- 当前仓库到底实现了什么
- 哪些部分是当前主线
- 哪些部分只是保留的旧 LoRS baseline
- 哪些公式已经实现
- 哪些地方当前代码仍是简化版、surrogate 版或仅保留接口

如果某个环节在代码里只是近似实现、surrogate 实现、兼容实现或尚未完全闭环，本文会明确标注。

---

# 1. 项目概述

当前仓库同时包含两条线。

## 1.1 保留的旧 LoRS baseline

旧 LoRS baseline 仍在仓库中保留，核心文件包括：

- `distill_tesla_lors.py`
- `buffer.py`
- `evaluate_only.py`
- `src/similarity_mining.py`
- `src/vl_distill_utils.py`

这条线的核心是：

- trajectory matching / synthetic distillation
- 构造 synthetic images / synthetic text embeddings / similarity matrix
- 用这些 synthetic data 评估检索模型

它当前仍可作为兼容 baseline 使用，但**已经不是当前仓库的主要研究主线**。

## 1.2 当前主要方法主线

当前仓库的新主线是：

- 多模态
- pair-aware
- 真实样本子集选择

输入是原始训练集中的真实图文对：

\[
\mathcal{D} = \{(v_i, t_i)\}_{i=1}^N
\]

输出是原始训练集中的真实 pair-level 子集索引：

- `selected_indices.json`

这里的 `selected_indices` 对应的是**真实 `(image, caption)` pair 的 `sample_idx`**，不是：

- synthetic images
- synthetic texts
- 单模态图像 coreset
- 单模态文本 coreset

也就是说，当前主线的最终目标是：

1. 从真实训练集里选出一个真实图文对子集
2. 再用这个真实子集训练检索模型
3. 在 Flickr30K / MS-COCO 上做 retrieval evaluation

---

# 2. 当前主线总流程

当前主线流程如下：

```text
train pairs
-> pair-level indexing
-> selection feature cache
-> image/text topology construction
-> collapse analysis
-> cross-modal correction
-> unified topology B*
-> unified spectral embedding
-> proxy optimization
-> subset matching
-> selected_indices
-> subset dataloader
-> retrieval train/eval
```

下面按步骤说明每一步的输入、输出、代码文件和落盘文件。

## 2.1 Pair-level indexing

**作用**

- 把训练样本单位固定为 `(image, caption)` pair
- 为每个真实 pair 分配唯一 `sample_idx`

**输入**

- Flickr30K / COCO train json

**输出**

- pair-level `sample_idx`
- `img_id`
- `caption`
- `image`

**对应代码**

- `data/flickr30k_dataset.py`
- `data/coco_dataset.py`
- `data/subset_dataset.py`

**关键接口**

- `flickr30k_train.get_pair_metadata(sample_idx)`
- `flickr30k_train.get_sample(index, return_sample_idx=None)`
- `coco_train.get_pair_metadata(sample_idx)`
- `coco_train.get_sample(index, return_sample_idx=None)`
- `PairSubsetDataset`

**当前实现状态**

- 已实现
- `sample_idx` 是 pair-level 唯一索引，不是 `img_id`

## 2.2 Selection feature cache

**作用**

- 为 selection stage 缓存固定图像表示和固定文本表示

**输入**

- 原始 train split 图像
- train captions

**输出**

- `img_features_selection.pt`
- `txt_features_selection.pt`
- `sample_meta.json`
- `feature_info.json`

**入口**

- `run_feature_cache.py`

**核心代码**

- `src/feature_cache.py`
- `src/fixed_image_features.py`
- `src/fixed_text_features.py`

**当前实现状态**

- 已实现
- 默认已经是“selection stage 去视觉网络化”版本
- 但保留了 legacy fallback：
  - 通过 `--disable_selection_only_fixed_repr` 可以回到旧的视觉 backbone feature cache 路线

## 2.3 Image / Text topology construction

**作用**

- 基于 selection-stage features 分别构建 image graph 与 text graph

**输入**

- `img_features_selection.pt`
- `txt_features_selection.pt`
- `sample_meta.json`

**输出**

按模态分别写到：

- `artifacts/topology_graph/{dataset}/train/{model_tag}/{modality}/{graph_tag}/`

典型文件：

- `knn_indices.pt`
- `knn_distances.pt`
- `local_scale.pt`
- `directed_graph.npz`
- `A_directed.npz`
- `symmetric_graph.npz`
- `adjacency.npz`
- `B_graph.npz`
- `transition_graph.npz`
- `laplacian_normalized.npz`
- `L_sym.npz`
- `eigenvalues.pt`
- `eigenvectors.pt`（如果启用）
- `spectral_embedding.pt`（如果启用）
- `summary.json`

**入口**

- `run_topology_graph.py`

**核心代码**

- `src/topology_graph.py`

**当前实现状态**

- 已实现
- 已有：
  - graph-only PCA / random projection
  - `rho`
  - `sigma` 二分搜索
  - directed fuzzy graph
  - fuzzy union 对称化
  - multi-scale graph merge
  - optional MST connectivity
  - normalized Laplacian
  - eigenspectrum
  - collapse metrics

## 2.4 Collapse analysis + cross-modal correction + unified topology

**作用**

- 比较 image / text 拓扑健康度
- 用健康模态校正坍缩模态
- 融合得到 unified topology `B*`
- 从 `B*` 提取 unified spectral embedding

**输入**

- `B^I`
- `B^T`
- 两个模态各自的 `summary.json`

**输出**

目录：

- `artifacts/cross_modal_topology/{dataset}/train/{model_tag}/{fusion_tag}/`

典型文件：

- `healthy_graph.npz`
- `healthy_transition.npz`
- `collapsed_graph.npz`
- `corrected_graph_directed.npz`
- `corrected_graph_symmetric.npz`
- `unified_graph.npz`
- `B_star.npz`
- `unified_transition.npz`
- `unified_laplacian_sym.npz`
- `L_star.npz`
- `unified_first_eigvals.npy`
- `unified_eigvecs.npy`（如果启用）
- `unified_spectral_embedding.npy`
- `V_full_multi.npy`
- `sample_meta.json`
- `modality_selection.json`
- `summary.json`

**入口**

- `run_cross_modal_topology.py`

**核心代码**

- `src/cross_modal_topology.py`

**当前实现状态**

- 已实现
- 默认融合方式是：
  - 健康模态图与校正后的坍缩模态图的**乘法交集式融合**
- 当前只支持：
  - `fusion_mode=intersection`

## 2.5 Proxy optimization + subset matching

**作用**

- 在统一表示 / 统一谱嵌入空间里优化连续代理点
- 再把代理点匹配回真实 pair-level 样本

**输入**

- selection-stage feature cache
- unified topology `B*`
- unified spectral embedding `V_full_multi`

**输出**

目录：

- `artifacts/subset_selection/{dataset}/train/{model_tag}/ratio_xx/{method}/`

典型文件：

- `selected_indices.json`
- `selected_meta.json`
- `matched_proxy_meta.json`
- `summary.json`
- `proxy_points.pt`
- `matching_cost.pt`

**入口**

- `run_subset_selection.py`

**核心代码**

- `src/subset_match.py`
- `src/proxy_optimization.py`

**当前实现状态**

- `ours_baseline`：已实现
- `ours_full`：已实现
- 但要注意：
  - optimization loop 中的 `lambda_match` 和 `lambda_graph` 当前使用的是 surrogate reference loss
  - 离散 matching 后的精确 `match_loss / graph_loss` 当前是**事后计算并记录在 diagnostics 中**，不是直接 end-to-end 回传到 proxy optimization

## 2.6 Subset dataloader + retrieval train/eval

**作用**

- 根据 `selected_indices` 恢复真实子集 dataloader
- 用真实子集训练 retrieval 模型
- 在 val/test 上评测

**输入**

- `selected_indices.json`
- 原始 train dataset

**输出**

目录：

- `artifacts/subset_train/{dataset}/{model_tag}/ratio_xx/{subset_tag}/seed_x/`

典型文件：

- `best_checkpoint.pt`
- `metrics.json`
- `history.json`
- `train.log`

**入口**

- `run_subset_train.py`

**核心代码**

- `src/subset_train.py`
- `src/epoch.py`
- `src/networks.py`

**当前实现状态**

- 已实现
- 支持：
  - `NFNet + BERT`
  - `ResNet-50 + BERT`
  - `ViT-B/16 + BERT`

---

# 3. Selection stage 与 training/eval stage 的严格边界

这是当前代码主线里最重要的设计边界。

## 3.1 Selection stage

当前 selection stage 的约束是：

- **不能使用视觉神经网络**
- image side 使用固定、非神经网络表示
- text side 使用固定 BERT embedding

用于：

- feature cache for selection
- topology graph construction
- collapse analysis
- cross-modal correction
- unified topology
- unified spectral embedding
- proxy optimization
- subset matching

当前 selection stage 默认图像表示：

- `hog_color`

备选图像表示：

- `raw_pca`

当前 selection stage 文本表示：

- 固定 `BERT`
- 只离线编码缓存
- 不在 selection stage 中参与参数更新

## 3.2 Training / Evaluation stage

这一阶段可以使用视觉网络。

当前支持：

- `NFNet + BERT`
- `ResNet-50 + BERT`
- `ViT-B/16 + BERT`

这一阶段只负责：

- 读取 `selected_indices`
- 恢复真实子集
- 训练 retrieval 模型
- 输出检索指标

## 3.3 为什么要这样分开

当前代码的边界设计是：

- selection-stage 的图像表示
  - 尽量固定、非神经
  - 目的是降低对某个视觉 backbone 的偏置
- training-stage 的图像表示
  - 可以是具体 backbone
  - 目的是验证选出的真实子集对不同 backbone 的泛化

因此：

- selection-stage 的图像表示
- training-stage 的视觉 backbone 表示

**不是同一回事**。

---

# 4. 数据处理全流程

## 4.1 支持的数据集

当前主线支持：

- `Flickr30K`
- `MS-COCO`

对应数据模块：

- `data/flickr30k_dataset.py`
- `data/coco_dataset.py`

## 4.2 样本单位

当前样本单位是：

- pair-level `(image, caption)` pair

不是：

- image-level
- img_id-level

当前 `sample_idx` 的含义：

- 唯一对应一条 annotation
- 即一个真实 `(image, caption)` pair

## 4.3 dataset 返回字段

### train dataset 默认行为

当 `return_sample_idx=False`（默认）时：

- Flickr：
  - `(image, caption, img_id)`
- COCO：
  - `(image, caption, img_id)`

### train dataset 开启 pair-level 索引

当 `return_sample_idx=True` 时：

- Flickr：
  - `(image, caption, sample_idx, img_id)`
- COCO：
  - `(image, caption, sample_idx, img_id)`

### 相关代码

- `data/flickr30k_dataset.py`
- `data/coco_dataset.py`
- `data/__init__.py`

## 4.4 真实子集如何恢复

当前通过：

- `data/subset_dataset.py`

里的 `PairSubsetDataset` 恢复真实子集。

`selected_indices.json` 的内容格式是：

```json
{
  "selected_indices": [12, 105, 238]
}
```

`PairSubsetDataset` 会把这些 pair-level `sample_idx` 映射回原始 train dataset 的真实样本。

## 4.5 Selection-stage feature cache

### image side

- 从原始图像文件读取
- 提取固定非神经表示
- 缓存为 `img_features_selection.pt`

### text side

- 从 caption 文本读取
- 用固定 BERT 编码
- 缓存为 `txt_features_selection.pt`

### sample_meta

`sample_meta.json` 中当前至少保存：

- `sample_idx`
- `img_id`
- `dataset`
- `split`
- `caption`
- `image`
- `raw_image_id`

## 4.6 Training-stage 数据流

流程如下：

1. 读取 `selected_indices.json`
2. 用 `PairSubsetDataset` 包装 train dataset
3. 构造真实子集 train loader
4. val/test 仍使用原始 retrieval eval dataset
5. 调用 retrieval train/eval

---

# 5. 图像与文本表示：当前代码到底怎么做

## 5.1 图像表示（selection stage）

当前 selection stage 图像表示实现于：

- `src/fixed_image_features.py`

并由：

- `src/feature_cache.py`

调用。

当前支持两种 fixed image representation。

### A. `hog_color`（默认）

**代码函数**

- `extract_hog_color_features(...)`
- `extract_fixed_image_features(..., method="hog_color")`

**具体流程**

1. 读取图像，转 `RGB`
2. resize 到固定大小
3. 提取 HOG 特征
4. 提取颜色直方图
5. 拼接成一个固定长度向量
6. 做 L2 normalize

**HOG 代表什么**

- HOG = Histogram of Oriented Gradients
- 它刻画局部边缘方向和纹理结构

**颜色直方图代表什么**

- 统计颜色分布
- 当前支持：
  - `rgb`
  - `hsv`

**当前默认参数**

来自 `run_feature_cache.py`：

- `selection_image_size = 128`
- `hog_orientations = 9`
- `hog_pixels_per_cell = 8`
- `hog_cells_per_block = 2`
- `color_hist_bins = 16`
- `color_space = rgb`

**代码当前实现状态**

- 已实现
- 依赖 `scikit-image`
- 不是深度网络表示

### B. `raw_pca`

**代码函数**

- `_extract_raw_pixel_vector(...)`
- `extract_raw_pca_features(...)`
- `extract_fixed_image_features(..., method="raw_pca")`

**具体流程**

1. 图像 resize 到较小分辨率
2. flatten 成 raw pixel 向量
3. 用 `IncrementalPCA` 拟合降维
4. transform 后做 L2 normalize

**当前默认参数**

来自 `run_feature_cache.py`：

- `selection_raw_resize_size = 32`
- `selection_raw_pca_dim = 256`

**PCA 在哪里做**

- 在 `src/fixed_image_features.py` 里用 `sklearn.decomposition.IncrementalPCA`

**代码当前实现状态**

- 已实现
- 是图像固定表示，不用视觉神经网络

## 5.2 文本表示（selection stage）

当前 selection stage 文本表示实现于：

- `src/fixed_text_features.py`

调用链：

- `extract_fixed_text_features(...)`
- 内部实例化 `src.networks.TextEncoder`
- 当前只允许 `text_repr_method="bert"`

**当前含义**

- selection stage 文本侧使用固定 BERT 向量
- 在 feature cache 阶段一次性离线编码并缓存
- 不在 selection stage 中参与参数更新

**缓存文件**

- `txt_features_selection.pt`

## 5.3 training/eval stage 的模型表示

这部分与 selection stage 明确区分。

当前 training/eval stage 使用：

- `NFNet + BERT`
- `ResNet-50 + BERT`
- `ViT-B/16 + BERT`

对应代码：

- `run_subset_train.py`
- `src/subset_train.py`
- `src/networks.py`

**重要边界**

这些视觉 backbone：

- 不参与 selection-stage feature cache
- 不参与 selection-stage topology construction
- 不参与 selection-stage subset selection

它们只用于后面的 retrieval train/eval。

---

# 6. 单模态拓扑构建：公式与当前代码实现

当前单模态建图的核心实现位于：

- `src/topology_graph.py`

入口是：

- `run_topology_graph.py`

下面区分“理想公式”和“当前代码实现”。

## 6.1 Graph features

设原始 selection-stage 特征为：

- image：
  \[
  x_i^I \in \mathbb{R}^{d_I}
  \]
- text：
  \[
  x_i^T \in \mathbb{R}^{d_T}
  \]

建图前可做 graph-only reduction，得到：

\[
g_i^I \in \mathbb{R}^{d_g}, \quad g_i^T \in \mathbb{R}^{d_g}
\]

### 当前代码实现

对应函数：

- `preprocess_features_for_knn(...)`
- `reduce_graph_features(...)`

支持：

- `none`
- `pca`
- `random_projection`

默认：

- `graph_reduce_method = pca`
- `graph_feature_dim = 256`

**实现状态**

- 已实现

## 6.2 kNN graph

对每个模态，当前先在 graph features 上做 kNN。

对应函数：

- `compute_knn_sklearn(...)`
- `compute_knn_faiss(...)`
- `compute_knn(...)`

支持 backend：

- `sklearn`
- `faiss`
- `auto`

支持 metric：

- `euclidean`
- `cosine`

### 当前代码实现状态

- 已实现
- `faiss` 是可选依赖
- `auto` 会优先尝试 `faiss`

## 6.3 局部连接距离 \(\rho_i\)

理想公式：

\[
\rho_i = \min_{j \in N_k(i), j \neq i} d(g_i, g_j)
\]

### 当前代码实现

对应函数：

- `compute_rho(distances, local_connectivity=1.0)`

当前实现不是纯粹“最小非零邻居距离”的硬编码，而是带 `local_connectivity` 的 UMAP 风格插值版本：

- `local_connectivity = 1.0` 时，等价于取第一个最近邻距离
- 其他值时允许插值

默认：

- `local_connectivity = 1.0`

**实现状态**

- 已实现

## 6.4 局部尺度 \(\sigma_i\)

理想公式：

\[
\sum_{j=1}^{k} \exp\left(-\frac{\max(0, d(g_i,g_{i_j}) - \rho_i)}{\sigma_i}\right) = \log_2(k)
\]

### 当前代码实现

对应函数：

- `_sigma_objective(...)`
- `compute_sigmas(...)`

当前实现：

- 用二分搜索求解每个点的 `sigma_i`
- 如果未手动指定 `bandwidth`，默认 target 不是 `log2(k)`，而是：

\[
\log_2(k + 1)
\]

也就是说，**当前代码和理想公式有一个常数项差异**：

- 理想写法：`log2(k)`
- 当前代码：`log2(k+1)`

这是 README 里必须明确的实现细节。

**实现状态**

- 已实现
- 但 target 采用的是 `log2(k+1)` 版本

## 6.5 有向边权 \(A_{ij}\)

理想公式：

\[
A_{ij} =
\exp\left(
-\frac{\max(0, d(g_i,g_j)-\rho_i)}{\sigma_i}
\right)
\]

若 \(j \notin N_k(i)\)，则 \(A_{ij}=0\)

### 当前代码实现

对应函数：

- `build_directed_graph(...)`

当前实现和上式一致：

- 只对 kNN 邻居赋值
- 再 clip 到 `[0,1]`
- 存成 sparse CSR

**实现状态**

- 已实现

## 6.6 fuzzy union 对称化 \(B_{ij}\)

理想公式：

\[
B_{ij} = A_{ij} + A_{ji} - A_{ij}A_{ji}
\]

### 当前代码实现

对应函数：

- `symmetrize_graph(...)`

实现方式：

- `sym = A + A^T - A ⊙ A^T`

与理想 fuzzy union 公式一致。

**实现状态**

- 已实现

## 6.7 Multi-scale k 融合

理想公式：

\[
B_{ij}^{multi} = 1 - \prod_{\ell}(1 - B_{ij}^{(k_\ell)})
\]

### 当前代码实现

对应函数：

- `parse_multi_scale_ks(...)`
- `build_multiscale_fuzzy_graph(...)`
- `fuzzy_union_merge(...)`

当前代码支持：

- `merge_mode = union`
- `merge_mode = mean`
- `merge_mode = max`

其中：

- `union` 最接近上面的 fuzzy-union 多尺度融合

默认：

- `multi_scale_merge_mode = union`

**实现状态**

- 已实现
- 但 README 需要说明：
  - 代码除了 `union` 之外，还支持 `mean/max` 两种工程 merge

## 6.8 MST 连通增强

理想思路：

- 若图不连通，则在参考距离图上加 MST 边

### 当前代码实现

对应函数：

- `add_mst_connectivity(...)`

当前实现流程：

1. 从参考局部距离图构造 `minimum_spanning_tree`
2. 把 MST 距离转成权重：

\[
\exp(-d / \bar{\sigma}) \cdot \text{weight_scale}
\]

3. 再和原图做 `maximum`

这与理想的 “\(B \leftarrow \max(B, M^{mst})\)” 方向一致，但 MST 权重是用当前代码里的指数映射构造的，不是直接原样拷贝距离。

**实现状态**

- 已实现
- 为可选开关

## 6.9 对称归一化 Laplacian

理想公式：

\[
L_{sym} = I - D^{-1/2} B D^{-1/2}
\]

### 当前代码实现

对应函数：

- `build_laplacian(graph, normalized=True)`
- `build_graph_artifacts(...)`

底层用：

- `scipy.sparse.csgraph.laplacian(..., normed=True)`

**实现状态**

- 已实现

## 6.10 谱嵌入

当前单模态 graph artifact 中已经支持：

- eigvals
- eigvecs
- spectral embedding

对应函数：

- `compute_spectrum(...)`
- `build_spectral_embedding(...)`
- `build_graph_artifacts(...)`

当前实现：

- 小图：`numpy.linalg.eigh`
- 大图：`scipy.sparse.linalg.eigsh`
- 默认会丢掉第一个平凡特征向量，再截取 `embedding_dim`

**实现状态**

- 已实现

---

# 7. Collapse analysis / cross-modal correction / unified topology

当前实现位于：

- `src/cross_modal_topology.py`

## 7.1 raw image graph \(B^I\) 与 raw text graph \(B^T\)

来源：

- 分别从 `artifacts/topology_graph/.../image/...`
- 和 `artifacts/topology_graph/.../text/...`

读取：

- `symmetric_graph.npz`
- `transition_graph.npz`
- `summary.json`

在代码里：

- `load_graph_bundle(...)`

## 7.2 collapse score / SCI

当前代码没有单独命名为 `SCI`，但已经有 collapse 指标。

来源：

- `src/topology_graph.py`
  - `compute_collapse_metrics(...)`

当前计算方式：

1. 取非平凡特征值
2. 归一化成权重
3. 计算归一化谱熵
4. 定义：

\[
\text{collapse\_score} = 1 - \text{spectral\_entropy}
\]

所以：

- collapse score 越小，模态越健康
- spectral entropy 越高，模态越健康

**实现状态**

- 已实现
- 当前并未单独实现另一个独立 SCI 公式；主判断依据是 `collapse_score`

## 7.3 healthy modality 判定

对应函数：

- `choose_healthy_modality(...)`

当前规则：

1. 比较 `collapse_score`
2. 更小者为健康模态
3. 若相同，则比较 `spectral_entropy`
4. 更高者为健康模态

也可通过参数：

- `--prefer_healthy_modality`

手动指定。

**实现状态**

- 已实现

## 7.4 随机游走矩阵

理想公式：

\[
P(j|i) = \frac{B_{ij}}{\sum_k B_{ik}}
\]

### 当前代码实现

对应函数：

- `row_normalize_graph(...)`

在 `load_graph_bundle` 里直接读取了 topology stage 已保存的：

- `transition_graph.npz`

**实现状态**

- 已实现

## 7.5 跨模态校正

当前默认设健康模态为 `health`，坍缩模态为 `collapsed`。

代码函数：

- `correct_collapsed_graph(...)`

当前公式对应关系：

先把健康模态 transition 做元素幂：

\[
P_{health}^{(\alpha)}
\]

然后：

\[
B'_{collapsed} = B_{collapsed} \odot P_{health}^{(\alpha)}
\]

代码中具体为：

1. `sparse_elementwise_power(healthy_transition, alpha)`
2. `collapsed_graph.multiply(powered_transition)`
3. 再做一次 `fuzzy_union_symmetrize`

也就是说，当前实现和你给出的：

\[
B'^{T}_{ij} = B^T_{ij} \cdot (P^I(j|i))^\alpha
\]

是一致方向的，但代码是先做稀疏逐元素乘，再统一 fuzzy union 对称化。

**实现状态**

- 已实现

## 7.6 unified topology \(B^*\)

当前默认统一拓扑构造函数：

- `unify_topology(...)`

当前只支持：

- `fusion_mode="intersection"`

实现方式：

\[
B^* = B_{health} \odot B'_{collapsed}
\]

然后再做：

- `fuzzy_union_symmetrize`

所以当前默认 unified topology 是：

- 乘法交集融合
- 再对称化

**实现状态**

- 已实现
- weighted-sum 融合当前没有实现

## 7.7 unified Laplacian \(L^*\)

对应函数：

- `build_unified_spectral_artifacts(...)`

当前实现：

- 直接对 `unified_graph` 调用 `build_laplacian(..., normalized=True)`

即：

\[
L^* = I - (D^*)^{-1/2} B^* (D^*)^{-1/2}
\]

**实现状态**

- 已实现

---

# 8. Unified spectral embedding

当前 unified spectral embedding 位于：

- `src/cross_modal_topology.py`

## 8.1 构造方式

当前代码流程：

1. 得到 unified topology `B*`
2. 构造 normalized Laplacian `L*`
3. 做谱分解
4. 调用 `build_spectral_embedding(...)`

对应函数：

- `build_unified_spectral_artifacts(...)`
- `build_spectral_embedding(...)`

## 8.2 维度

由命令行参数控制：

- `--spectral_embedding_dim`

默认：

- `32`

因此：

\[
V_{full\_multi} \in \mathbb{R}^{N \times d_s}, \quad d_s = 32 \text{ (default)}
\]

## 8.3 是否取最小非零特征值对应向量

当前代码确实会：

- 优先取最小特征值方向
- 默认丢掉第一个平凡特征向量
- 再截取 `embedding_dim`

因此可以近似理解为：

- 取最小非平凡特征值对应特征向量构成谱嵌入

## 8.4 当前实现是 exact 还是 approximate

当前实现不是统一的 exact dense eigendecomposition：

- 小图：`numpy.linalg.eigh`
- 大图：`scipy.sparse.linalg.eigsh`

因此更准确地说：

- 小规模时是 dense eigh
- 大规模时是 sparse eigsh 近似最小特征值方向

## 8.5 输出文件

- `unified_spectral_embedding.npy`
- `V_full_multi.npy`

## 8.6 下游消费模块

当前下游直接消费 `V_full_multi` 的模块是：

- `src/subset_match.py`
  - `load_unified_artifacts(...)`
  - `build_reference_embedding(...)`

当前 `reference_embedding_mode` 支持：

- `concat`
- `spectral`
- `hybrid`

其中：

- `spectral`：只用 `V_full_multi`
- `hybrid`：原始 unified representation 与 `V_full_multi` 拼接

---

# 9. Proxy optimization：详细公式与当前实现

当前实现位于：

- `src/proxy_optimization.py`

由：

- `src/subset_match.py`
 里的 `run_proxy_optimized_selection(...)`

调用。

## 9.1 代理点定义

当前代码中：

\[
Y = \{y_r\}_{r=1}^M
\]

其中：

\[
M = \max(1, \text{round}(N \cdot ratio))
\]

对应函数：

- `compute_subset_size(...)`

**实现状态**

- 已实现

## 9.2 当前主要优化空间

当前 proxy optimization 的输入不是直接原始 image/text 特征，而是：

1. 先构造 unified representation：
   - `build_unified_representation(...)`
   - 当前为 image/text L2 normalize 后的 concat
2. 再构造 reference embedding：
   - `build_reference_embedding(...)`
   - 可以是：
     - `concat`
     - `spectral`
     - `hybrid`

因此当前代理点优化空间取决于：

- `reference_embedding_mode`

默认：

- `hybrid`

也就是说，**当前默认并不是只在 `V_full_multi` 上优化**，而是：

- 原始 unified representation
- 与 unified spectral embedding 的混合空间

这是当前实现和理想“仅在 unified spectral embedding 上优化”的一个重要差异。

## 9.3 经验特征函数

当前实现函数：

- `compute_empirical_characteristic_components(...)`
- `compute_characteristic_function(...)`

定义为：

\[
\phi_U(\omega) = \frac{1}{n} \sum_a \exp(i \omega^\top u_a)
\]

代码里采用的是稳定的实部/虚部分开计算：

- `cos`
- `sin`

不直接用 Python complex tensor 做损失。

**实现状态**

- 已实现

## 9.4 CFD

当前实现函数：

- `cfd_loss(...)`
- `frequency_alignment_loss(...)`

当前代码对应：

\[
L_{CFD} = \sum_\omega |\phi_{proxy}(\omega) - \phi_{full}(\omega)|^2
\]

实现上是：

- 实部平方差 + 虚部平方差

目前没有显式支持任意 `w(\omega)`，等价于统一权重。

**实现状态**

- 已实现
- 当前频率权重默认是均匀的

## 9.5 PD-CFD

当前实现函数：

- `pd_cfd_loss(...)`

当前代码已经实现：

1. CFD 主项
2. phase-aware 项

具体代码逻辑：

- 先计算：
  - `A = sqrt(real^2 + imag^2 + eps)`
  - `theta = atan2(imag, real)`
- 再计算：

\[
\min(A_{full}, A_{proxy}) \cdot (1 - \cos(\theta_{proxy} - \theta_{full}))
\]

并乘：

- `lambda_phase`

这和你要求的默认版本是一致的。

当前 `phase_weights` 支持：

- `uniform`
- `linear`

但整体仍然是工程实现，不是单独分离的两阶段优化器。

**实现状态**

- 已实现

## 9.6 DPP / diversity loss

当前实现函数：

- `build_rbf_kernel(...)`
- `dpp_diversity_loss(...)`

当前优先使用的是：

\[
L_{div} = - \log \det(K_Y + \epsilon I)
\]

其中：

\[
K_Y(r,s) = \exp\left(-\frac{\|y_r-y_s\|^2}{2\sigma^2}\right)
\]

若 `slogdet` 符号异常，则直接返回 `0.0`，没有额外 fallback 到 pairwise repulsion regularizer。

所以当前状态是：

- logdet DPP 实现已存在
- 显式 alt repulsion loss 没有单独实现成另一路

**实现状态**

- 已实现（logdet 版本）
- fallback 只是返回 0，不是单独的 `L_div_alt`

## 9.7 Matching loss 与 graph loss：当前优化环节到底怎么接

这是当前代码最需要诚实说明的地方。

### 理想公式

理想上希望：

\[
L_{match} = \frac{1}{M}\sum_r \|y_r - v_{\pi(r)}\|^2
\]

\[
L_{graph} = \mathrm{Tr}(Y^\top L_{sub} Y)
\]

### 当前代码实际情况

在 `optimize_proxy_points(...)` 内，当前使用的是 surrogate：

- `match_loss = nearest_reference_loss(proxy_points, match_reference)`
- `graph_loss = nearest_reference_loss(proxy_points, graph_reference)`

这里：

- `match_reference` 通常是 reference embedding 本身
- `graph_reference` 是 unified graph 上一跳 context 构造的 reference

也就是说，**optimization loop 内的 `lambda_match / lambda_graph` 还不是离散匹配后的精确项**。

### 精确项在哪里

离散 matching 完成后，当前代码会显式计算：

- `compute_match_loss(...)`
- `compute_graph_regularization(...)`

并把它们写入：

- `matching_debug.json`
- `summary`

所以当前状态是：

- optimization loop 中：surrogate
- discrete matching 后：精确值被事后计算并记录
- 还没有做 end-to-end differentiable coupling

**实现状态**

- surrogate 版：已实现
- matching 后精确 diagnostics：已实现
- 完整 end-to-end 精确联合优化：未实现

## 9.8 总目标

当前 `optimize_proxy_points(...)` 中的真实损失是：

\[
L_{total}
= L_{main}
  + \lambda_{reg} L_{reg}
  + \lambda_{div} L_{div}
  + \lambda_{match} L_{match}^{surrogate}
  + \lambda_{graph} L_{graph}^{surrogate}
\]

其中：

- `L_main`
  - `cfd` 或 `pd_cfd`
- `L_reg`
  - 代理点对初始化点的二范数正则
- `L_div`
  - DPP logdet
- `L_match^{surrogate}`
  - nearest reference loss
- `L_graph^{surrogate}`
  - nearest graph reference loss

因此，当前代码**不是**纯粹：

\[
L_{main} + \lambda_{div}L_{div} + \lambda_{match}L_{match} + \lambda_{graph}L_{graph}
\]

而是多了一个：

- `reg_weight * reg_loss`

并且 `match / graph` 在优化时是 surrogate 版本。

## 9.9 PDAS

当前实现函数：

- `sample_frequency_pool(...)`
- `schedule_pdas_frequencies(...)`
- `sample_pdas_frequencies(...)`

当前 `use_pdas=True` 时：

1. 先采样 `Omega_pool`
2. 根据当前 step 得到：

\[
\tau_t = \tau_{min} + (\tau_{max} - \tau_{min}) \cdot t/T
\]

3. 先筛：

\[
\|\omega\| \le \tau_t
\]

4. 再按 discrepancy score 选 top-k：

\[
|\phi_{proxy}(\omega) - \phi_{full}(\omega)|^2
\]

这已经是一个 discrepancy-aware 的 low-to-high scheduler。

另外还保留了：

- `uniform` schedule mode

**实现状态**

- 已实现
- 当前是实用化版本，不是单独论文模块式封装

---

# 10. Subset matching：详细公式与当前实现

当前实现位于：

- `src/subset_match.py`

## 10.1 当前匹配代价

### 方式 A：candidate_topk（默认）

当前默认不是直接构造完整 \(M \times N\) 全局矩阵，而是：

1. 先对每个代理点找 top-k 候选真实节点
2. 再在候选集合上构造代价

对应函数：

- `compute_candidate_neighbors(...)`
- `build_candidate_costs(...)`
- `compute_proxy_sample_costs(...)`

代价形式为：

\[
C(r,i) =
\lambda_{geom}\|y_r - v_i\|^2
 \lambda_{topo}\|y_r - \tilde{v}_i\|^2
- w_{deg}\cdot score(i)
\]

这里：

- `geometry_cost`
- `topology_cost`
- `graph_scores`

都被显式分开保存。

这不是你理想公式里的纯 degree-aware cost，而是当前仓库原有 graph-aware cost 的延续版本。

### 方式 B：degree_aware_global

当前代码也支持更贴近公式的全局度加权代价：

对应函数：

- `build_degree_aware_cost_matrix(...)`

公式为：

\[
C_{geom}(r,i)=\|y_r-v_i\|^2
\]

\[
C(r,i)=\frac{C_{geom}(r,i)}{\deg_{B^*}(i)+\epsilon}
\]

这是当前代码里更直接对应你要求公式的版本。

## 10.2 Hungarian matching

当前代码**确实使用了**：

- `scipy.optimize.linear_sum_assignment`

对应函数：

- `run_hungarian_matching(...)`
- `run_degree_aware_global_matching(...)`

当前有两种模式：

1. `candidate_topk`
   - 在候选池上做 Hungarian / conflict resolution
2. `degree_aware_global`
   - 在完整 `M x N` 代价矩阵上做全局 Hungarian

## 10.3 selected_indices 如何导出

`run_subset_selection(...)` 中：

1. 获取 matching 结果
2. `sort_selected_indices(...)`
3. 调用 `save_selection_outputs(...)`

最终写出：

- `selected_indices.json`

其中长度为：

\[
\max(1, \text{round}(N \cdot subset\_ratio))
\]

当前会保证唯一性，因为 Hungarian 本身就是一对一匹配。

## 10.4 matching diagnostics

当前会输出：

- `matching_cost.pt`
- `matching_debug.json`
- `matched_proxy_meta.json`

`matching_debug.json` 当前可能包含：

- `assignment_mode`
- `duplicate_resolution_rounds`
- `local_hungarian_calls`
- `hungarian_rows`
- `match_loss`
- `graph_loss`

---

# 11. Subset train / retrieval eval

当前下游训练评测实现位于：

- `run_subset_train.py`
- `src/subset_train.py`
- `src/epoch.py`
- `src/networks.py`

## 11.1 输入

输入包括：

- `selected_indices.json`
- 原始 train dataset
- val/test retrieval eval dataset

## 11.2 支持的模型

当前训练阶段支持：

- `NFNet + BERT`
- `ResNet-50 + BERT`
- `ViT-B/16 + BERT`

具体通过：

- `CLIPModel_full`

完成。

## 11.3 指标

最终输出指标包括：

- `Image-to-Text Retrieval: R@1 / R@5 / R@10`
- `Text-to-Image Retrieval: R@1 / R@5 / R@10`
- `Mean Recall`

代码里：

- `epoch_test(...)`
- `itm_eval(...)`

然后在 `src/subset_train.py` 里转换为：

- `i2t_r1`
- `i2t_r5`
- `i2t_r10`
- `t2i_r1`
- `t2i_r5`
- `t2i_r10`
- `mean_recall`

## 11.4 指标与 checkpoint 保存

输出目录：

- `artifacts/subset_train/{dataset}/{backbone}_bert/ratio_xx/{subset_tag}/seed_x/`

当前保存：

- `best_checkpoint.pt`
- `metrics.json`
- `history.json`
- `train.log`

`metrics.json` 至少包含：

- `dataset`
- `backbone`
- `text_encoder`
- `subset_ratio`
- `subset_size`
- `seed`
- `best_epoch`
- `i2t_r1`
- `i2t_r5`
- `i2t_r10`
- `t2i_r1`
- `t2i_r5`
- `t2i_r10`
- `mean_recall`
- `val_mean_recall`
- `val_metrics`
- `test_metrics`

---

# 12. 命令行参数总表

这里按模块列出当前关键参数。参数名以当前代码为准。

## 12.1 Feature cache：`run_feature_cache.py`

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset` | 必填 | `flickr` / `coco` |
| `--image_encoder` | 必填 | 主要用于 cache 目录命名及 legacy 模式 |
| `--text_encoder` | `bert` | selection-stage text encoder |
| `--selection_image_repr_method` | `hog_color` | 固定图像表示方法 |
| `--selection_text_repr_method` | `bert` | 固定文本表示方法 |
| `--selection_image_size` | `128` | `hog_color` resize 大小 |
| `--selection_raw_resize_size` | `32` | `raw_pca` resize 大小 |
| `--selection_raw_pca_dim` | `256` | `raw_pca` PCA 维度 |
| `--selection_image_batch_size` | `512` | 图像 fixed repr 批大小 |
| `--selection_text_batch_size` | `256` | 文本固定 BERT 批大小 |
| `--hog_orientations` | `9` | HOG 参数 |
| `--hog_pixels_per_cell` | `8` | HOG 参数 |
| `--hog_cells_per_block` | `2` | HOG 参数 |
| `--color_hist_bins` | `16` | 颜色直方图 bins |
| `--color_space` | `rgb` | `rgb` / `hsv` |
| `--disable_selection_only_fixed_repr` | False | 若启用则回到 legacy DNN cache 路线 |
| `--image_root` | `None` | 图片根目录 |
| `--ann_root` | `data/Flickr30k_ann` | 标注目录 |
| `--cache_root` | `artifacts/feature_cache` | 缓存目录 |
| `--batch_size` | `64` | legacy 模型特征缓存时使用 |
| `--num_workers` | `4` | DataLoader worker |
| `--device` | auto | 设备 |
| `--overwrite` | False | 是否重建缓存 |

**注意**

`run_feature_cache.py` 自身的默认 `image_root` 仍是旧的大写路径：

- `data/Flickr30k`
- `data/COCO`

而 `scripts/experiments/common.sh` 已改成：

- `data/flickr30k`
- `data/coco`

因此如果你手动运行 `run_feature_cache.py`，建议显式传 `--image_root`。

## 12.2 Topology graph：`run_topology_graph.py`

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset` | 必填 | 数据集 |
| `--modality` | 必填 | `image` / `text` |
| `--feature_cache_root` | `artifacts/feature_cache` | selection cache 根目录 |
| `--output_root` | `artifacts/topology_graph` | 输出目录 |
| `--metric` | `euclidean` | 距离度量 |
| `--k` | `15` | kNN k 值 |
| `--knn_k` | `None` | `k` 的别名 |
| `--multi_scale_ks` | `None` | 多尺度 k，逗号分隔 |
| `--multi_scale_merge_mode` | `union` | `union / mean / max` |
| `--use_mst_connectivity` | False | 是否加 MST 连通增强 |
| `--mst_weight_scale` | `1.0` | MST 权重比例 |
| `--num_eigs` | `32` | 提取特征值数量 |
| `--spectral_embedding_dim` | `32` | 谱嵌入维度 |
| `--n_jobs` | `None` | sklearn kNN 并行数 |
| `--knn_backend` | `auto` | `auto / sklearn / faiss` |
| `--faiss_use_gpu` | False | 是否用 GPU FAISS |
| `--graph_reduce_method` | `pca` | `none / pca / random_projection` |
| `--graph_feature_dim` | `256` | graph-only reduction 维度 |
| `--local_connectivity` | `1.0` | 计算 `rho` 的局部连接参数 |
| `--bandwidth` | `None` | 若设置则覆盖默认 target |
| `--sigma_search_steps` | `64` | `sigma` 二分搜索步数 |

## 12.3 Cross-modal topology：`run_cross_modal_topology.py`

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset` | 必填 | 数据集 |
| `--topology_root` | `artifacts/topology_graph` | 单模态图根目录 |
| `--output_root` | `artifacts/cross_modal_topology` | 输出目录 |
| `--metric` | `euclidean` | 主 metric 标签 |
| `--image_metric` | `None` | image graph metric |
| `--text_metric` | `None` | text graph metric |
| `--k` | `15` | graph tag 中使用 |
| `--multi_scale_ks` | `None` | 多尺度图 tag |
| `--alpha` | `1.0` | 健康模态校正强度 |
| `--fusion_mode` | `intersection` | 当前只支持 `intersection` |
| `--prefer_healthy_modality` | `None` | 手动指定健康模态 |
| `--num_eigs` | `64` | unified 图谱分解数量 |
| `--spectral_embedding_dim` | `32` | unified spectral dim |

## 12.4 Subset selection：`run_subset_selection.py`

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset` | 必填 | 数据集 |
| `--feature_cache_root` | `artifacts/feature_cache` | feature cache 根目录 |
| `--cross_modal_root` | `artifacts/cross_modal_topology` | unified topology 根目录 |
| `--output_root` | `artifacts/subset_selection` | 输出目录 |
| `--budget_ratio` | 必填 | `0.05 / 0.1 / 0.2` |
| `--representation_mode` | `concat` | 当前 unified repr 构造方式 |
| `--reference_embedding_mode` | `hybrid` | `concat / spectral / hybrid` |
| `--spectral_weight` | `1.0` | hybrid 模式中谱嵌入权重 |
| `--selection_method` | `proxy_opt` | `baseline / proxy_opt` |
| `--degree_weight` | `0.1` | graph score tie-break / cost weighting |
| `--geometry_weight` | `1.0` | 几何代价权重 |
| `--proxy_projection_dim` | `128` | proxy 空间随机投影维度 |
| `--proxy_init_method` | `kmeans` | `kmeans / minibatch_kmeans / sample` |
| `--proxy_objective_mode` | `pd_cfd` | `cfd / pd_cfd` |
| `--use_pdcfd` | False | 若开启则强制 `pd_cfd` |
| `--proxy_num_frequencies` | `64` | 每步使用的频率数 |
| `--proxy_frequency_scale` | `1.0` | 频率尺度 |
| `--proxy_lr` | `0.05` | 学习率 |
| `--proxy_num_steps` | `200` | 优化步数 |
| `--proxy_reg_weight` | `0.01` | 初始化点正则 |
| `--use_pdas` | False | 是否启用 PDAS |
| `--pdas_num_stages` | `4` | PDAS 阶段数 |
| `--pdas_schedule_mode` | `low_to_high` | `low_to_high / uniform` |
| `--num_freq_pool` | `256` | PDAS 频率池大小 |
| `--tau_min` | `0.1` | PDAS 最小半径 |
| `--tau_max` | `1.0` | PDAS 最大半径 |
| `--use_dpp` | False | 是否启用 DPP diversity |
| `--lambda_div` | `0.01` | diversity 权重 |
| `--lambda_match` | `0.05` | surrogate match 权重 |
| `--lambda_graph` | `0.05` | surrogate graph 权重 |
| `--lambda_phase` | `0.1` | PD-CFD phase 项权重 |
| `--diversity_sigma` | `1.0` | DPP RBF sigma |
| `--matching_top_k` | `64` | candidate-topk matching 候选数 |
| `--matching_candidate_batch_size` | `128` | candidate cost 批处理 |
| `--matching_cost_mode` | `candidate_topk` | `candidate_topk / degree_aware_global` |
| `--topology_weight` | `0.5` | topology cost 权重 |
| `--topology_hop_weight` | `0.5` | graph reference 构造权重 |

## 12.5 Subset train：`run_subset_train.py`

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `--dataset` | 必填 | 数据集 |
| `--image_root` | 必填 | 图片根目录 |
| `--ann_root` | 必填 | 标注根目录 |
| `--selected_indices_path` | 必填 | 真实子集索引 |
| `--output_root` | `artifacts/subset_train` | 输出目录 |
| `--subset_ratio` | 必填 | 子集比例 |
| `--subset_tag` | `None` | 结果目录标签 |
| `--image_encoder` | 必填 | `nfnet / resnet50 / resnet-50 / vit_b16 / vit-b16 / vit-b/16` |
| `--text_encoder` | `bert` | 文本编码器 |
| `--image_size` | `224` | 输入尺寸 |
| `--batch_size_train` | `64` | train batch |
| `--batch_size_test` | `128` | test batch |
| `--text_batch_size` | `1024` | text eval batch |
| `--epochs` | `20` | 训练轮数 |
| `--eval_interval` | `1` | 验证间隔 |
| `--lr_teacher_img` | `0.1` | image lr |
| `--lr_teacher_txt` | `0.1` | text projection lr |
| `--momentum` | `0.9` | SGD momentum |
| `--weight_decay` | `5e-4` | weight decay |
| `--lr_decay_gamma` | `0.1` | lr decay |
| `--seed` | `0` | 随机种子 |
| `--device` | `None` | 自动选择设备 |
| `--no_aug` | False | 是否禁用增强 |

---

# 13. 输出目录与文件命名规范

## 13.1 Feature cache

目录：

```text
artifacts/feature_cache/{dataset}/train/{image_encoder}_{text_encoder}/
```

selection-stage 默认输出：

- `img_features_selection.pt`
- `txt_features_selection.pt`
- `sample_meta.json`
- `feature_info.json`

legacy 模式可能输出：

- `img_features.pt`
- `txt_features.pt`

## 13.2 Topology graph

目录：

```text
artifacts/topology_graph/{dataset}/train/{model_tag}/{modality}/{graph_tag}/
```

典型文件：

- `knn_indices.pt`
- `knn_distances.pt`
- `local_scale.pt`
- `directed_graph.npz`
- `A_directed.npz`
- `symmetric_graph.npz`
- `adjacency.npz`
- `B_graph.npz`
- `transition_graph.npz`
- `laplacian_normalized.npz`
- `L_sym.npz`
- `eigenvalues.pt`
- `eigenvectors.pt`
- `spectral_embedding.pt`
- `sample_meta.json`
- `summary.json`

## 13.3 Cross-modal topology

目录：

```text
artifacts/cross_modal_topology/{dataset}/train/{model_tag}/k{k}_{metric}_a{alpha}/
```

典型文件：

- `healthy_graph.npz`
- `healthy_transition.npz`
- `collapsed_graph.npz`
- `corrected_graph_directed.npz`
- `corrected_graph_symmetric.npz`
- `unified_graph.npz`
- `B_star.npz`
- `unified_transition.npz`
- `unified_laplacian_sym.npz`
- `L_star.npz`
- `unified_first_eigvals.npy`
- `unified_eigvecs.npy`
- `unified_spectral_embedding.npy`
- `V_full_multi.npy`
- `modality_selection.json`
- `sample_meta.json`
- `summary.json`

## 13.4 Subset selection

目录：

```text
artifacts/subset_selection/{dataset}/train/{model_tag}/ratio_xx/{method}/
```

其中：

- `ours_baseline` 对应 `baseline`
- `ours_full` 对应 `proxy_opt`

典型文件：

- `selected_indices.json`
- `selected_meta.json`
- `matched_proxy_meta.json`
- `summary.json`
- `proxy_points.pt`
- `proxy_init.pt`
- `proxy_debug.json`
- `projection_matrix.pt`
- `frequency_points.pt`
- `initial_frequency_points.pt`
- `matching_cost.pt`
- `matching_debug.json`

## 13.5 Subset train

目录：

```text
artifacts/subset_train/{dataset}/{model_tag}/ratio_xx/{subset_tag}/seed_x/
```

典型文件：

- `best_checkpoint.pt`
- `metrics.json`
- `history.json`
- `train.log`

---

# 14. 当前实现状态总结

## 14.1 已实现

- pair-level `sample_idx`
- `selected_indices -> PairSubsetDataset -> retrieval train/eval`
- selection-stage fixed image representation：
  - `hog_color`
  - `raw_pca`
- selection-stage fixed BERT text representation
- 单模态 topology graph：
  - kNN
  - `rho`
  - `sigma` 二分搜索
  - directed fuzzy graph
  - fuzzy union
  - optional multi-scale merge
  - optional MST
  - normalized Laplacian
  - spectrum
  - collapse score
- cross-modal correction：
  - healthy modality selection
  - random-walk reweighting
  - intersection-style unified topology
- unified spectral embedding
- `ours_baseline`
- `ours_full`
- CFD
- PD-CFD
- PDAS
- DPP logdet diversity
- Hungarian matching
- degree-aware global matching
- retrieval train/eval with：
  - `NFNet + BERT`
  - `ResNet-50 + BERT`
  - `ViT-B/16 + BERT`

## 14.2 部分实现 / surrogate 实现

- `lambda_match` / `lambda_graph`
  - 在 proxy optimization 内当前是 surrogate nearest-reference 形式
  - 不是离散 matching 后的精确项直接反传
- `fusion_mode`
  - 当前只实现了 `intersection`
  - weighted-sum 融合未实现
- DPP fallback
  - 当前没有显式 `L_div_alt`
  - 数值异常时直接返回 0

## 14.3 仍待验证

- full Flickr30K / full COCO 的长时间服务器实跑稳定性
- `faiss` backend 在实际服务器上的收益
- `ours_full` 在多 seed、多 backbone 下的完整主表结果
- `Random / K-center / Herding / LoRS` 统一纳入新实验脚本的完整闭环

## 14.4 保留 baseline

保留但不是主线：

- `distill_tesla_lors.py`
- `buffer.py`
- `evaluate_only.py`
- `src/similarity_mining.py`
- `src/vl_distill_utils.py`

## 14.5 当前最小可运行链路

当前最小主线链路是：

- `Flickr30K`
- selection image repr = `hog_color`
- selection text repr = `bert`
- method = `ours_full`
- subset ratio = `5%`
- train backbone = `nfnet`

---

# 15. 最小运行示例

## 15.1 分步运行

### 1. selection feature cache

```bash
python run_feature_cache.py \
  --dataset flickr \
  --image_encoder nfnet \
  --text_encoder bert \
  --selection_image_repr_method hog_color \
  --selection_text_repr_method bert \
  --image_root data/flickr30k \
  --ann_root data/Flickr30k_ann \
  --cache_root artifacts/feature_cache \
  --selection_image_size 128 \
  --overwrite
```

### 2. image topology

```bash
python run_topology_graph.py \
  --dataset flickr \
  --split train \
  --image_encoder nfnet \
  --text_encoder bert \
  --modality image \
  --feature_cache_root artifacts/feature_cache \
  --output_root artifacts/topology_graph \
  --metric euclidean \
  --knn_k 15 \
  --graph_reduce_method pca \
  --graph_feature_dim 256 \
  --num_eigs 32 \
  --spectral_embedding_dim 32
```

### 3. text topology

```bash
python run_topology_graph.py \
  --dataset flickr \
  --split train \
  --image_encoder nfnet \
  --text_encoder bert \
  --modality text \
  --feature_cache_root artifacts/feature_cache \
  --output_root artifacts/topology_graph \
  --metric cosine \
  --knn_k 15 \
  --graph_reduce_method pca \
  --graph_feature_dim 256 \
  --num_eigs 32 \
  --spectral_embedding_dim 32
```

### 4. unified topology

```bash
python run_cross_modal_topology.py \
  --dataset flickr \
  --split train \
  --image_encoder nfnet \
  --text_encoder bert \
  --topology_root artifacts/topology_graph \
  --output_root artifacts/cross_modal_topology \
  --image_metric euclidean \
  --text_metric cosine \
  --k 15 \
  --alpha 1.0 \
  --num_eigs 64 \
  --spectral_embedding_dim 32
```

### 5. subset selection

```bash
python run_subset_selection.py \
  --dataset flickr \
  --split train \
  --image_encoder nfnet \
  --text_encoder bert \
  --feature_cache_root artifacts/feature_cache \
  --cross_modal_root artifacts/cross_modal_topology \
  --output_root artifacts/subset_selection \
  --metric euclidean \
  --k 15 \
  --alpha 1.0 \
  --budget_ratio 0.05 \
  --selection_method proxy_opt \
  --reference_embedding_mode hybrid \
  --use_pdcfd \
  --use_pdas \
  --use_dpp \
  --device cuda
```

### 6. subset train / eval

```bash
python run_subset_train.py \
  --dataset flickr \
  --image_root data/flickr30k \
  --ann_root data/Flickr30k_ann \
  --selected_indices_path artifacts/subset_selection/flickr/train/nfnet_bert/ratio_05/proxy_opt/selected_indices.json \
  --subset_ratio 0.05 \
  --subset_tag ours_full \
  --image_encoder nfnet \
  --text_encoder bert \
  --output_root artifacts/subset_train \
  --epochs 20 \
  --seed 0 \
  --device cuda \
  --no_aug
```

## 15.2 一步运行

当前也可以直接使用实验脚本：

```bash
bash scripts/experiments/run_pipeline.sh flickr nfnet 0.05 0 ours_full
```

这个脚本会串起：

1. feature cache
2. image topology
3. text topology
4. cross-modal topology
5. subset selection
6. subset training

**当前脚本真实状态**

- 已支持：
  - `ours_baseline`
  - `ours_full`
- 还未支持通过统一新脚本直接跑：
  - `Random`
  - `K-center / Herding`
  - `LoRS`

---

# 16. 与旧 LoRS baseline 的关系

当前仓库没有删除旧 LoRS baseline。

旧 baseline 相关文件：

- `distill_tesla_lors.py`
- `buffer.py`
- `evaluate_only.py`
- `src/similarity_mining.py`
- `src/vl_distill_utils.py`

这些文件主要仍服务于：

- synthetic distillation
- replay buffer
- synthetic evaluation

它们和当前多模态真实子集选择主线是并存关系，不是统一实现。

因此在阅读仓库时，建议把它们看成：

- 兼容保留的 baseline 代码

而把下面这组文件看成当前主线：

- `run_feature_cache.py`
- `run_topology_graph.py`
- `run_cross_modal_topology.py`
- `run_subset_selection.py`
- `run_subset_train.py`
- `src/feature_cache.py`
- `src/fixed_image_features.py`
- `src/fixed_text_features.py`
- `src/topology_graph.py`
- `src/cross_modal_topology.py`
- `src/proxy_optimization.py`
- `src/subset_match.py`
- `src/subset_train.py`
