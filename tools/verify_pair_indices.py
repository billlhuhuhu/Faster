import argparse
import os
import sys

from torchvision import transforms

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data.coco_dataset import coco_train
from data.flickr30k_dataset import flickr30k_train
from data.subset_dataset import PairSubsetDataset, load_selected_indices, save_selected_indices


def build_dataset(args):
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor()])
    dataset_cls = {
        "flickr": flickr30k_train,
        "coco": coco_train,
    }[args.dataset]
    return dataset_cls(
        transform=transform,
        image_root=args.image_root,
        ann_root=args.ann_root,
        return_sample_idx=True,
    )


def parse_selected_indices(args):
    if args.selected_indices_file:
        return load_selected_indices(args.selected_indices_file)
    if args.selected_indices:
        return [int(x.strip()) for x in args.selected_indices.split(",") if x.strip()]
    return list(range(args.verify_count))


def verify_dataset(dataset, verify_count):
    print(f"Dataset size: {len(dataset)}")
    for sample_idx in range(min(verify_count, len(dataset))):
        meta = dataset.get_pair_metadata(sample_idx)
        assert meta["sample_idx"] == sample_idx, f"sample_idx mismatch at {sample_idx}"
    print(f"Verified stable pair metadata for the first {min(verify_count, len(dataset))} samples.")


def verify_subset(dataset, selected_indices, cache_path=None):
    subset = PairSubsetDataset(dataset, selected_indices, return_sample_idx=True)
    for local_idx, sample_idx in enumerate(selected_indices):
        subset_meta = subset.get_pair_metadata(local_idx)
        base_meta = dataset.get_pair_metadata(sample_idx)
        assert subset_meta == base_meta, f"subset recovery mismatch for sample_idx={sample_idx}"

    if cache_path:
        save_selected_indices(cache_path, selected_indices)
        restored = load_selected_indices(cache_path)
        assert restored == [int(x) for x in selected_indices], "cached selected indices mismatch"
        print(f"Saved and reloaded selected indices at {cache_path}")

    print(f"Verified subset recovery for {len(selected_indices)} selected pairs.")


def maybe_verify_image_loading(dataset, selected_indices):
    if not selected_indices:
        return

    first_meta = dataset.get_pair_metadata(selected_indices[0])
    image_path = os.path.join(dataset.image_root, first_meta["image"])
    if not os.path.exists(image_path):
        print(f"Skip image-loading check because image file is missing: {image_path}")
        return

    _, caption, sample_idx, img_id = dataset.get_sample(selected_indices[0], return_sample_idx=True)
    print("Loaded one real sample successfully:")
    print(f"  sample_idx={sample_idx}, img_id={img_id}, caption={caption[:80]}")


def main():
    parser = argparse.ArgumentParser(description="Verify pair-level sample indices and subset recovery.")
    parser.add_argument("--dataset", choices=["flickr", "coco"], required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--ann_root", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--verify_count", type=int, default=5)
    parser.add_argument("--selected_indices", type=str, default="0,1,2")
    parser.add_argument("--selected_indices_file", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default="artifacts/debug/selected_indices.json")
    args = parser.parse_args()

    dataset = build_dataset(args)
    selected_indices = parse_selected_indices(args)

    verify_dataset(dataset, args.verify_count)
    verify_subset(dataset, selected_indices, cache_path=args.cache_path)
    maybe_verify_image_loading(dataset, selected_indices)

    print("All checks passed.")


if __name__ == "__main__":
    main()
