# Flickr30K 数据集下载指南

## 方式一: 使用批处理脚本 (Windows推荐)

双击运行项目根目录下的 `download_flickr30k.bat` 文件即可开始下载。

## 方式二: 使用Python脚本

在项目根目录下执行:

```bash
# 下载所有数据集 (train + val + test)
python tools/download_flickr30k.py --split all --max_workers 10

# 只下载训练集
python tools/download_flickr30k.py --split train --max_workers 10

# 只下载验证集
python tools/download_flickr30k.py --split val --max_workers 10

# 只下载测试集
python tools/download_flickr30k.py --split test --max_workers 10
```

## 参数说明

- `--split`: 指定下载哪个数据集，可选值: `train`, `val`, `test`, `all`
- `--max_workers`: 并发下载数量，默认为10，可根据网络情况调整
- `--ann_root`: 标注文件目录，默认为 `data/Flickr30k_ann`
- `--image_root`: 图片保存目录，默认为 `data/Flickr30k`

## 数据集说明

Flickr30K 数据集包含:

- **训练集**: 29,783张图片，145,000个描述
- **验证集**: 1,014张图片
- **测试集**: 1,000张图片

图片下载位置: `data/Flickr30k/flickr30k-images/`

## 注意事项

1. **首次下载**: 完整数据集约13GB，下载时间取决于网络速度
2. **断点续传**: 脚本支持断点续传，如果中断可以重新运行继续下载
3. **已有文件**: 已下载的文件会自动跳过
4. **网络问题**: 如果下载失败，可以重试，脚本会自动重试3次

## 手动下载 (可选)

如果脚本下载遇到问题，也可以手动从以下地址下载:

- 基础URL: `https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_images/`

图片文件名格式: `{image_id}.jpg`

将下载的图片保存到 `data/Flickr30k/flickr30k-images/` 目录下即可。
