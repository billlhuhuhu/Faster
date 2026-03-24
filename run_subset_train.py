import argparse

from src.subset_train import train_and_evaluate_subset


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate retrieval on a selected real subset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--ann_root", type=str, required=True)
    parser.add_argument("--selected_indices_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/subset_train")
    parser.add_argument("--subset_ratio", type=float, required=True, choices=[0.05, 0.1, 0.2])
    parser.add_argument("--subset_tag", type=str, default=None)

    parser.add_argument("--image_encoder", type=str, required=True, choices=["nfnet", "resnet50", "resnet-50", "vit_b16", "vit-b16", "vit-b/16"])
    parser.add_argument("--text_encoder", type=str, default="bert", choices=["bert"])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size_train", type=int, default=64)
    parser.add_argument("--batch_size_test", type=int, default=128)
    parser.add_argument("--text_batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1)

    parser.add_argument("--lr_teacher_img", type=float, default=0.1)
    parser.add_argument("--lr_teacher_txt", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_aug", action="store_true", default=False)

    parser.add_argument("--image_pretrained", type=bool, default=True)
    parser.add_argument("--text_pretrained", type=bool, default=True)
    parser.add_argument("--image_trainable", type=bool, default=True)
    parser.add_argument("--text_trainable", type=bool, default=False)
    parser.add_argument("--only_has_image_projection", type=bool, default=False)
    parser.add_argument("--distill", type=bool, default=False)
    parser.add_argument("--loss_type", type=str, default="InfoNCE")
    return parser


def main():
    args = build_parser().parse_args()
    outputs = train_and_evaluate_subset(args)
    print("Subset training finished:")
    print(f"  output_dir: {outputs['output_dir']}")
    print(f"  checkpoint_path: {outputs['checkpoint_path']}")
    print(f"  metrics_path: {outputs['metrics_path']}")
    print(f"  log_path: {outputs['log_path']}")
    print(f"  subset_size: {outputs['subset_size']}")
    print(f"  mean_recall: {outputs['metrics']['mean_recall']:.2f}")


if __name__ == "__main__":
    main()
