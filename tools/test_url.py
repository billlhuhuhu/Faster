import requests

# 测试图片URL
test_url = "https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_images/1000092795.jpg"

print(f"测试URL: {test_url}")
print()

try:
    response = requests.head(test_url, timeout=10)
    print(f"状态码: {response.status_code}")
    print(f"可访问: {'是' if response.status_code == 200 else '否'}")

    if response.status_code == 200:
        print("\n尝试下载前1KB测试...")
        response = requests.get(test_url, timeout=10, stream=True)
        content = next(response.iter_content(1024))
        print(f"成功获取内容，大小: {len(content)} bytes")
        print("图片URL可以正常访问!")
    else:
        print("尝试备用URL...")

        # 尝试其他可能的URL
        alt_urls = [
            f"https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k/1000092795.jpg",
            f"https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k-images/1000092795.jpg",
        ]

        for alt_url in alt_urls:
            print(f"\n测试备用URL: {alt_url}")
            try:
                response = requests.head(alt_url, timeout=10)
                print(f"状态码: {response.status_code}")
                if response.status_code == 200:
                    print("此URL可以访问!")
                    break
            except Exception as e:
                print(f"错误: {e}")

except Exception as e:
    print(f"错误: {e}")
