# LoRS_Distill

本仓库最初基于 LoRS 代码框架搭建，现在同时包含两条并行主线：

- 原始的 **LoRS 多模态数据蒸馏 baseline**
- 新增的 **真实样本子集选择与检索评测主线**

新主线的目标不是继续做轨迹匹配蒸馏，而是直接从真实的 `(image, caption)` pair 中选出代表性子集，并在该真实子集上训练图文检索模型。


## 一、项目当前状态

目前新主线已经按阶段完成了从“真实样本索引”到“训练与评测入口”的完整工程搭建，整体流程已经串起来了：

1. 训练样本 `pair-level` 唯一索引
2. 全量图像/文本特征提取与缓存
3. 单模态拓扑建图与坍缩分析
4. 跨模态拓扑校正与统一拓扑重建
5. 简单可运行的真实子集选择 baseline
6. 完整方法版：频域分布对齐 + 连续代理点优化 + 拓扑约束匹配
7. 真实子集训练与检索评测入口
8. 第一版实验矩阵、配置文件和服务器脚本模板

原始 LoRS 的代码和入口仍然保留，没有故意破坏旧 baseline 的可运行性。


## 二、目前已经实现了什么

### 1. 真实 pair 的唯一索引

Flickr30K 和 MS-COCO 的训练集现在支持可选的 `pair-level sample_idx`：

- `data/flickr30k_dataset.py`
- `data/coco_dataset.py`
- `data/subset_dataset.py`

默认行为保持不变。只有在 `return_sample_idx=True` 时，训练集才会返回：

```python
(image, caption, sample_idx, img_id)
```

其中：

- `sample_idx` 对应唯一的 `(image, caption)` pair
- `img_id` 仍然保留旧逻辑


### 2. 全量特征缓存

入口：

- `run_feature_cache.py`

核心模块：

- `src/feature_cache.py`

当前支持：

- 图像编码器：
  - `nfnet`
  - `resnet-50`
  - `vit-b/16`
- 文本编码器：
  - `bert`

输出内容：

- `img_features.pt`
- `txt_features.pt`
- `sample_meta.json`
- `feature_info.json`


### 3. 单模态拓扑建图与坍缩分析

入口：

- `run_topology_graph.py`

核心模块：

- `src/topology_graph.py`

已实现：

- 基于缓存特征构建 kNN 图
- 支持欧氏距离和余弦距离
- `rho / sigma` 局部尺度估计
- 有向模糊图
- 对称图
- 转移图
- 归一化拉普拉斯
- 谱分析
- 基于谱熵的稳定坍缩指标


### 4. 跨模态拓扑校正与统一拓扑重建

入口：

- `run_cross_modal_topology.py`

核心模块：

- `src/cross_modal_topology.py`

已实现：

- 健康模态判定
- 用健康模态修正坍缩模态
- 统一拓扑 `B*` 重建
- 中间图与摘要结果落盘


### 5. 真实子集选择

入口：

- `run_subset_selection.py`

核心模块：

- `src/subset_match.py`
- `src/proxy_optimization.py`

目前有两条可运行方法：

- `baseline`
  - 统一表示
  - K-means 代理中心
  - 最近真实样本映射
  - 图中心性 tie-break
- `proxy_opt`
  - 经验特征函数对齐
  - 连续代理点优化
  - 拓扑感知匹配代价
  - 匈牙利式冲突消解

输出包括：

- `selected_indices.json`
- `selected_meta.json`
- `summary.json`
- `proxy_points.pt`
- `matching_cost.pt`


### 6. 真实子集训练与检索评测

入口：

- `run_subset_train.py`

核心模块：

- `src/subset_train.py`

复用了以下工具层：

- `src/epoch.py`
- `src/networks.py`

已实现：

- 从 `selected_indices` 构造真实子集 dataloader
- 在真实子集上训练 retrieval 模型
- 在 val/test 上做图文检索评测
- 保存 checkpoint、metrics、history、log

当前评测指标：

- Image-to-Text Retrieval：`R@1 / R@5 / R@10`
- Text-to-Image Retrieval：`R@1 / R@5 / R@10`
- `Mean Recall`


### 7. 第一版实验脚本与配置

配置文件：

- `configs/experiments/main_table.json`
- `configs/experiments/generalization_table.json`
- `configs/experiments/ablation_table.json`

服务器脚本模板：

- `scripts/experiments/common.sh`
- `scripts/experiments/run_pipeline.sh`
- `scripts/experiments/run_main_table.sh`
- `scripts/experiments/run_generalization_table.sh`
- `scripts/experiments/run_ablation_table.sh`


## 三、第一版固定实验目标

数据集：

- Flickr30K
- MS-COCO

主模型：

- NFNet + BERT

泛化模型：

- ResNet-50 + BERT
- ViT-B/16 + BERT

指标：

- Image-to-Text Retrieval：`R@1 / R@5 / R@10`
- Text-to-Image Retrieval：`R@1 / R@5 / R@10`
- Mean Recall

真实子集比例：

- `5%`
- `10%`
- `20%`


## 四、现在还差什么

虽然主方法的代码已经基本接起来了，但离“论文级完整实验”还有几步：

### 1. 服务器上的全量实跑

当前代码主线已经能跑，但还需要在服务器上用完整数据和 GPU 环境真正跑完：

- Flickr30K
- MS-COCO
- 多个 backbone
- 多个 seed


### 2. 额外 baseline 还没完全接入

实验矩阵里已经给这些 baseline 留好了位置，但目前还没全部做成统一入口：

- `Random`
- `K-center` 或 `Herding`
- `LoRS`

也就是说，目前在新主线里真正已经可跑的方法是：

- `ours_baseline`
- `ours_full`


### 3. 消融实验还需要补具体开关

消融矩阵已经设计好了，但这些版本还没有全部实现成命令行可切换的形式：

- `w/o_frequency_alignment`
- `w/o_topology_penalty`
- `w/o_cross_modal_correction`
- `nearest_neighbor_match`


### 4. 还缺结果汇总脚本

后面还需要一个轻量聚合脚本，用于：

- 多 seed 取均值和标准差
- 汇总主表/泛化表/消融表
- 导出 `csv/json`


### 5. 跨架构实验还没真正跑完

目前已经补了对以下 backbone 的兼容：

- `nfnet`
- `resnet-50`
- `vit-b/16`

但真正的跨架构结果还需要在服务器上完整跑出来。


## 五、新方法现在是怎么实现的

当前实现路径如下：

```text
真实训练 pair
-> 全量 image/text 特征缓存
-> image graph + text graph
-> 坍缩分析
-> 跨模态校正
-> unified topology B*
-> 代理点优化 / 子集匹配
-> selected_indices
-> subset dataloader
-> retrieval training + evaluation
```

完整方法版的主要逻辑是：

1. 对所有训练 pair 提取图像与文本特征
2. 在单模态空间中建图并分析模态坍缩
3. 用更健康的模态去修正更坍缩的模态
4. 重建统一拓扑图 `B*`
5. 通过频域目标优化连续代理点
6. 用带拓扑惩罚的代价把代理点映射回真实样本
7. 得到真实 `selected_indices`
8. 在真实子集上训练和评测检索模型


## 六、服务器推荐目录结构

代码目录建议：

```text
LoRS_Distill/
  data/
  src/
  configs/
  scripts/
  run_feature_cache.py
  run_topology_graph.py
  run_cross_modal_topology.py
  run_subset_selection.py
  run_subset_train.py
```

数据和权重建议在服务器上单独准备：

```text
data/
  Flickr30k/
  COCO/
  Flickr30k_ann/

distill_utils/
  checkpoints/
    bert-base-uncased/
    nfnet_l0_ra2-45c6688d.pth
```


## 七、上传代码时的注意事项

以下目录通常不需要上传到 GitHub：

- `artifacts/`
- `__pycache__/`
- `wandb/`
- `.tmp-conda/`

其中 `artifacts/` 是中间缓存和实验输出目录，建议在服务器上重新生成。


## 八、主入口脚本

特征缓存：

```bash
python run_feature_cache.py --help
```

单模态拓扑：

```bash
python run_topology_graph.py --help
```

跨模态拓扑：

```bash
python run_cross_modal_topology.py --help
```

子集选择：

```bash
python run_subset_selection.py --help
```

子集训练：

```bash
python run_subset_train.py --help
```

完整流水线模板：

```bash
bash scripts/experiments/run_pipeline.sh flickr nfnet 0.05 0 ours_full
```


## 九、旧的 LoRS baseline 仍然保留

原始 LoRS 相关入口仍然在：

- `buffer.py`
- `distill_tesla_lors.py`
- `evaluate_only.py`
- `sh/`

因此当前仓库实际上支持两条并行主线：

- 原始 LoRS 蒸馏 baseline
- 新的真实样本子集选择主线


## 十、引用

如果你使用原始 LoRS baseline，请引用 LoRS 原论文。
