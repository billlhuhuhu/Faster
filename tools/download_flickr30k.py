#!/usr/bin/env python
"""
下载完整的 Flickr30K 数据集图片到指定目录
支持断点续传
"""
import os
import urllib.request
import urllib.error
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import argparse
from tqdm import tqdm
import time

# 设置超时和重试
socket.setdefaulttimeout(60)

def download_image_with_retry(img_url, save_path, max_retries=3):
    """下载单个图片，支持重试"""
    for attempt in range(max_retries):
        try:
            # 创建临时文件
            temp_path = save_path + '.tmp'

            # 检查是否已有临时文件（断点续传）
            existing_size = 0
            if os.path.exists(temp_path):
                existing_size = os.path.getsize(temp_path)

            # 发起请求
            req = urllib.request.Request(img_url)
            if existing_size > 0:
                req.add_header('Range', f'bytes={existing_size}-')

            # 下载
            with urllib.request.urlopen(req) as response:
                total_size = int(response.getheader('Content-Length', 0)) + existing_size
                mode = 'ab' if existing_size > 0 else 'wb'

                with open(temp_path, mode) as f:
                    chunk_size = 8192
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)

            # 下载完成，重命名为正式文件
            if os.path.exists(save_path):
                os.remove(save_path)
            os.rename(temp_path, save_path)

            return True, save_path

        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False, f"{img_url}: 文件不存在 (404)"
            time.sleep(1)
        except Exception as e:
            time.sleep(2)
            continue

    return False, f"{img_url}: 达到最大重试次数"

def download_flickr30k_images(annotation_file, image_root, max_workers=10):
    """从标注文件下载所有图片"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 提取所有图片路径
    image_paths = set()
    if isinstance(data, list):
        for item in data:
            if 'image' in item:
                image_paths.add(item['image'])

    print(f"找到 {len(image_paths)} 张需要下载的图片")

    # 构建 URL 和保存路径
    base_url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_images/"

    tasks = []
    skipped = 0
    for img_path in image_paths:
        # 获取图片文件名
        img_filename = os.path.basename(img_path)
        # 构建下载URL
        img_url = f"{base_url}{img_filename}"

        save_path = os.path.join(image_root, img_path)
        # 如果文件已存在且大小大于0，跳过
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            skipped += 1
            continue

        tasks.append((img_url, save_path))

    print(f"需要下载: {len(tasks)} 张图片, 已存在: {skipped} 张")
    if len(tasks) == 0:
        print("所有图片已存在，无需下载")
        return

    # 使用多线程下载
    success_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image_with_retry, url, path): (url, path)
                  for url, path in tasks}

        for future in tqdm(as_completed(futures), total=len(tasks), desc="下载图片"):
            success, info = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                # 只打印部分失败信息，避免刷屏
                if fail_count <= 10:
                    tqdm.write(f"下载失败: {info}")

    print(f"\n下载完成! 成功: {success_count}, 失败: {fail_count}, 跳过: {skipped}")

def main():
    parser = argparse.ArgumentParser(description='下载Flickr30K数据集图片')
    parser.add_argument('--ann_root', type=str,
                        default='F:/Sinhu/experiments/Nips数据集压缩/LoRS_Distill/data/Flickr30k_ann',
                        help='标注文件目录')
    parser.add_argument('--image_root', type=str,
                        default='F:/Sinhu/experiments/Nips数据集压缩/LoRS_Distill/data/Flickr30k',
                        help='图片保存目录')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test', 'all'],
                        help='下载哪个数据集')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='并发下载数量')

    args = parser.parse_args()

    if args.split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits = [args.split]

    for split in splits:
        ann_file = os.path.join(args.ann_root, f'flickr30k_{split}.json')
        if os.path.exists(ann_file):
            print(f"\n{'='*60}")
            print(f"开始下载 {split} 数据集")
            print(f"{'='*60}")
            download_flickr30k_images(ann_file, args.image_root, args.max_workers)
        else:
            print(f"标注文件不存在: {ann_file}")

if __name__ == '__main__':
    main()
