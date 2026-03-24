#!/bin/bash
# 下载完整的 Flickr30K 数据集

ANNS_ROOT="data/Flickr30k_ann"
IMAGES_ROOT="data/Flickr30k"

# 创建输出目录
mkdir -p "$IMAGES_ROOT/flickr30k-images"

echo "========================================"
echo "开始下载 Flickr30K 完整数据集"
echo "========================================"
echo ""

# 从标注文件中提取所有图片ID并下载
for split in train val test; do
    echo "处理 $split 数据集..."
    python -c "
import json
import os

ann_file = '${ANNS_ROOT}/flickr30k_${split}.json'
image_root = '${IMAGES_ROOT}'

with open(ann_file, 'r') as f:
    data = json.load(f)

# 提取所有图片ID
image_ids = set()
for item in data:
    if 'image' in item:
        img_path = item['image']
        img_id = os.path.basename(img_path)
        image_ids.add(img_id)

# 写入下载列表
with open('${IMAGES_ROOT}/${split}_download_list.txt', 'w') as f:
    for img_id in sorted(image_ids):
        url = f'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_images/{img_id}'
        save_path = f'{image_root}/flickr30k-images/{img_id}'
        f.write(f'{url} -o {save_path}\n')

print(f'{split} 数据集: {len(image_ids)} 张图片')
print(f'下载列表已写入: ${IMAGES_ROOT}/${split}_download_list.txt')
"
done

echo ""
echo "========================================"
echo "生成下载列表完成!"
echo "========================================"
echo ""
echo "现在运行以下命令开始下载:"
echo "cd tools && python download_all_flickr30k.py"
