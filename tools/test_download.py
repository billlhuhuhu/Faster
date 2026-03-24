#!/usr/bin/env python
"""
测试下载一张图片
"""
import os
import urllib.request
import urllib.error

# 测试URL
test_url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_images/1000092795.jpg"
save_path = "F:/Sinhu/experiments/Nips数据集压缩/LoRS_Distill/data/Flickr30k/test_image.jpg"

print(f"测试URL: {test_url}")
print(f"保存路径: {save_path}")
print()

try:
    # 创建目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 下载
    print("开始下载...")
    urllib.request.urlretrieve(test_url, save_path)
    print("下载成功!")

    # 检查文件
    if os.path.exists(save_path):
        file_size = os.path.getsize(save_path)
        print(f"文件大小: {file_size} bytes ({file_size/1024:.2f} KB)")
        print("文件验证通过!")
    else:
        print("错误: 文件未找到")

except urllib.error.HTTPError as e:
    print(f"HTTP错误: {e.code} - {e.reason}")
    print("图片URL不可访问")

except Exception as e:
    print(f"错误: {e}")
