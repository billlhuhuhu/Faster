import json

# Check val and test files
for split in ['val', 'test']:
    ann_file = f'F:/Sinhu/experiments/Nips数据集压缩/LoRS_Distill/data/Flickr30k_ann/flickr30k_{split}.json'
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f'{split} - Total items:', len(data))
    print(f'{split} - First item keys:', data[0].keys() if data else 'empty')
    if data and 'image' in data[0]:
        print(f'{split} - First image path:', data[0]['image'])
        print(f'{split} - Image filename:', data[0]['image'].split('/')[-1])
    print()
