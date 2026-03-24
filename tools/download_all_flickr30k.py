#!/usr/bin/env python
"""
一键下载完整的 Flickr30K 数据集图片
"""
import os
import sys

def main():
    print("========================================")
    print("开始下载 Flickr30K 完整数据集")
    print("========================================")
    print()

    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    ann_root = os.path.join(project_root, 'data', 'Flickr30k_ann')
    image_root = os.path.join(project_root, 'data', 'Flickr30k')

    print(f"标注文件目录: {ann_root}")
    print(f"图片保存目录: {image_root}")
    print()

    # 导入下载脚本
    sys.path.insert(0, script_dir)
    from download_flickr30k import download_flickr30k_images

    # 下载 train, val, test 所有数据集
    splits = ['train', 'val', 'test']
    total_success = 0
    total_fail = 0

    for split in splits:
        ann_file = os.path.join(ann_root, f'flickr30k_{split}.json')
        if os.path.exists(ann_file):
            print(f"\n{'='*60}")
            print(f"开始下载 {split} 数据集")
            print(f"{'='*60}")
            print()
            download_flickr30k_images(ann_file, image_root, max_workers=10)
        else:
            print(f"标注文件不存在: {ann_file}")
            print(f"跳过 {split} 数据集")
        print()

    print("="*60)
    print("Flickr30K 数据集下载流程完成!")
    print("="*60)

if __name__ == '__main__':
    main()
