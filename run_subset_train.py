import os
import argparse

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from src.subset_train import train_and_evaluate_subset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}.")


def build_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate retrieval on a selected real subset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["flickr", "coco"])
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--ann_root", type=str, required=True)
    parser.add_argument("--selected_indices_path", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="artifacts/subset_train")
    budget_group = parser.add_mutually_exclusive_group(required=False)
    budget_group.add_argument("--subset_ratio", type=float, default=None)
    budget_group.add_argument("--subset_size", type=int, default=None)
    parser.add_argument("--subset_tag", type=str, default=None)
    parser.add_argument("--diagnostic_experiment_id", type=int, default=None)
    parser.add_argument("--enable_stage2_correction", dest="enable_stage2_correction", action="store_true", default=True)
    parser.add_argument("--disable_stage2_correction", dest="enable_stage2_correction", action="store_false")
    parser.add_argument("--enable_stage3_fusion", dest="enable_stage3_fusion", action="store_true", default=True)
    parser.add_argument("--disable_stage3_fusion", dest="enable_stage3_fusion", action="store_false")
    parser.add_argument("--enable_stage4_lsrc", dest="enable_stage4_lsrc", action="store_true", default=True)
    parser.add_argument("--disable_stage4_lsrc", dest="enable_stage4_lsrc", action="store_false")

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
    parser.add_argument("--enable_image_encoder_data_parallel", action="store_true", default=False)
    parser.add_argument("--image_encoder_data_parallel_device_ids", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_aug", action="store_true", default=False)

    parser.add_argument("--image_pretrained", type=str2bool, default=True)
    parser.add_argument("--text_pretrained", type=str2bool, default=True)
    parser.add_argument("--image_trainable", type=str2bool, default=True)
    parser.add_argument("--text_trainable", type=str2bool, default=False)
    parser.add_argument("--only_has_image_projection", type=str2bool, default=False)
    parser.add_argument("--distill", type=str2bool, default=False)
    parser.add_argument("--loss_type", type=str, default="InfoNCE")
    return parser


def main():
    args = build_parser().parse_args()
    exp_label = args.diagnostic_experiment_id if args.diagnostic_experiment_id is not None else "custom"
    print(f"[Experiment] Exp {exp_label}")
    print(f"  Stage2 Correction: {'ON' if args.enable_stage2_correction else 'OFF'}")
    print(f"  Stage3 Fusion: {'ON' if args.enable_stage3_fusion else 'OFF'}")
    print(f"  Stage4 LSRC: {'ON' if args.enable_stage4_lsrc else 'OFF'}")
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
