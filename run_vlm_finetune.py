import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.subset_dataset import load_selected_indices


def sanitize_name(value: str) -> str:
    return str(value).replace("\\", "-").replace("/", "-").replace(" ", "_")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Annotation file not found: {path_obj}")
    if path_obj.suffix.lower() == ".jsonl":
        records = []
        with path_obj.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    payload = json.loads(path_obj.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("data", "annotations", "samples"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        raise ValueError(f"Unsupported annotation dict format in {path_obj}; expected data/annotations/samples list.")
    if not isinstance(payload, list):
        raise ValueError(f"Unsupported annotation format in {path_obj}; expected list or jsonl.")
    return payload


def strip_image_token(text: str) -> str:
    return str(text).replace("<image>", "").strip()


def extract_llava_turn(record: Dict[str, Any]) -> Tuple[str, str]:
    conversations = record.get("conversations") or record.get("conversation") or record.get("messages")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("LLaVA record must contain a conversations/messages list with at least one user and assistant turn.")

    user_text = None
    assistant_text = None
    for item in conversations:
        role = str(item.get("from", item.get("role", ""))).lower()
        value = item.get("value", item.get("content", ""))
        if role in {"human", "user"} and user_text is None:
            user_text = strip_image_token(value)
        elif role in {"gpt", "assistant"} and user_text is not None:
            assistant_text = str(value).strip()
            break

    if user_text is None or assistant_text is None:
        raise ValueError("Could not find a human/user turn followed by a gpt/assistant answer.")
    return user_text, assistant_text


def resolve_image_path(record: Dict[str, Any], image_root: str) -> Path:
    image_value = record.get("image", record.get("image_path", record.get("file_name", "")))
    if not image_value:
        raise ValueError("LLaVA record does not contain an image/image_path/file_name field.")
    image_path = Path(str(image_value))
    if image_path.is_absolute():
        return image_path

    root = Path(image_root)
    direct_path = root / image_path
    if direct_path.exists():
        return direct_path

    candidates = []
    name = image_path.name
    parent = image_path.parent
    if name and not name.startswith("COCO_"):
        candidates.extend(
            [
                root / parent / f"COCO_train2014_{name}",
                root / parent / f"COCO_val2014_{name}",
                root / "train2014" / f"COCO_train2014_{name}",
                root / "val2014" / f"COCO_val2014_{name}",
            ]
        )
    candidates.extend(
        [
            root / "train2014" / image_path,
            root / "val2014" / image_path,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return direct_path


def build_qwen2vl_messages(image_path: Path, prompt: str, answer: Optional[str] = None) -> List[Dict[str, Any]]:
    content = [
        {"type": "image", "image": str(image_path)},
        {"type": "text", "text": prompt},
    ]
    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def select_records(
    records: Sequence[Dict[str, Any]],
    subset_mode: str,
    subset_ratio: Optional[float],
    selected_indices_path: Optional[str],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    total = len(records)
    if total <= 0:
        raise ValueError("No records found in annotation file.")

    subset_mode = str(subset_mode).lower()
    if subset_mode == "full":
        indices = list(range(total))
    elif subset_mode == "ours":
        if not selected_indices_path:
            raise ValueError("subset_mode=ours requires --selected_indices_path.")
        indices = load_selected_indices(selected_indices_path)
    elif subset_mode == "random":
        if subset_ratio is None:
            raise ValueError("subset_mode=random requires --subset_ratio as a percentage, e.g. 1, 5, 10.")
        count = max(1, int(round(total * float(subset_ratio) / 100.0)))
        rng = random.Random(int(seed))
        indices = sorted(rng.sample(range(total), min(count, total)))
    else:
        raise ValueError(f"Unsupported subset_mode: {subset_mode}")

    indices = [int(idx) for idx in indices if 0 <= int(idx) < total]
    if not indices:
        raise ValueError(f"Subset mode {subset_mode} produced an empty subset.")
    return [records[idx] for idx in indices], indices


class LlavaInstructionDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, Any]], image_root: str):
        self.records = list(records)
        self.image_root = str(image_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        prompt, answer = extract_llava_turn(record)
        image_path = resolve_image_path(record, self.image_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found for sample {index}: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return {
            "id": record.get("id", index),
            "image": image,
            "image_path": str(image_path),
            "prompt": prompt,
            "answer": answer,
        }


@dataclass
class Qwen2VlDataCollator:
    processor: Any
    max_length: int = 2048

    def _format_text(self, sample: Dict[str, Any], include_answer: bool) -> str:
        messages = build_qwen2vl_messages(
            Path(sample["image_path"]),
            sample["prompt"],
            sample["answer"] if include_answer else None,
        )
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        full_texts = [self._format_text(sample, include_answer=True) for sample in features]
        prompt_texts = [self._format_text(sample, include_answer=False) for sample in features]
        images = [sample["image"] for sample in features]

        batch = self.processor(
            text=full_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )
        prompt_batch = self.processor(
            text=prompt_texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.max_length),
        )

        labels = batch["input_ids"].clone()
        pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            labels[labels == int(pad_token_id)] = -100

        prompt_attention = prompt_batch.get("attention_mask")
        if prompt_attention is not None:
            prompt_lengths = prompt_attention.sum(dim=1).tolist()
        else:
            prompt_lengths = [prompt_batch["input_ids"].shape[1]] * len(features)
        for row_idx, prompt_len in enumerate(prompt_lengths):
            labels[row_idx, : min(int(prompt_len), labels.shape[1])] = -100
        batch["labels"] = labels
        return batch


def split_train_val(
    records: Sequence[Dict[str, Any]],
    val_records: Optional[Sequence[Dict[str, Any]]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(records)
    if val_records is not None:
        return records, list(val_records)
    if float(val_ratio) <= 0.0 or len(records) < 2:
        return records, []

    rng = random.Random(int(seed))
    indices = list(range(len(records)))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(records) * float(val_ratio))))
    val_count = min(val_count, max(1, len(records) - 1))
    val_set = set(indices[:val_count])
    train = [item for idx, item in enumerate(records) if idx not in val_set]
    val = [item for idx, item in enumerate(records) if idx in val_set]
    return train, val


def build_output_dir(args: argparse.Namespace, subset_size: int) -> Path:
    model_name = sanitize_name(args.model_name_or_path)
    dataset_name = sanitize_name(args.dataset_name)
    if args.subset_mode == "full":
        subset_tag = "full"
    else:
        subset_tag = f"{args.subset_mode}_{int(round(float(args.subset_ratio)))}"
    seed_tag = f"seed_{int(args.seed)}"
    return Path(args.output_root) / model_name / dataset_name / subset_tag / seed_tag


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_and_processor(args: argparse.Namespace):
    try:
        from transformers import AutoProcessor
        from transformers import BitsAndBytesConfig
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForVision2Seq
            model_cls = AutoModelForVision2Seq
    except ImportError as exc:
        raise ImportError(
            "Qwen2-VL finetuning requires transformers with Qwen2-VL support. "
            "Install/update transformers before running this experiment."
        ) from exc

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"

    quantization_config = None
    device_map = args.device_map
    if args.finetune_mode == "qlora":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        if not device_map:
            device_map = "auto"

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = model_cls.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map=device_map if device_map else None,
        trust_remote_code=bool(args.trust_remote_code),
    )
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.finetune_mode in {"lora", "qlora"}:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("LoRA/QLoRA mode requires peft. Install with: pip install peft") from exc
        if args.finetune_mode == "qlora":
            model = prepare_model_for_kbit_training(model)
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        lora_config = LoraConfig(
            r=int(args.lora_r),
            lora_alpha=int(args.lora_alpha),
            lora_dropout=float(args.lora_dropout),
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif args.finetune_mode != "full":
        raise ValueError(f"Unsupported finetune_mode: {args.finetune_mode}")
    return model, processor


def build_trainer(args: argparse.Namespace, model: Any, processor: Any, train_dataset: Dataset, eval_dataset: Optional[Dataset], output_dir: Path):
    try:
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise ImportError("Training requires transformers Trainer/TrainingArguments.") from exc

    eval_strategy_key = "evaluation_strategy"
    try:
        import inspect
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
            eval_strategy_key = "eval_strategy"
    except Exception:
        pass

    eval_strategy = "steps" if eval_dataset is not None and len(eval_dataset) > 0 else "no"
    save_strategy = "steps"
    training_kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(args.num_train_epochs),
        "per_device_train_batch_size": int(args.per_device_train_batch_size),
        "per_device_eval_batch_size": int(args.per_device_eval_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "warmup_ratio": float(args.warmup_ratio),
        "logging_steps": int(args.logging_steps),
        "save_steps": int(args.save_steps),
        "save_total_limit": int(args.save_total_limit),
        "save_strategy": save_strategy,
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "dataloader_num_workers": int(args.num_workers),
        "bf16": bool(args.bf16),
        "fp16": bool(args.fp16),
        "seed": int(args.seed),
        "load_best_model_at_end": bool(args.load_best_model_at_end and eval_strategy != "no"),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_dir": str(output_dir / "logs"),
    }
    training_kwargs[eval_strategy_key] = eval_strategy
    if eval_strategy != "no":
        training_kwargs["eval_steps"] = int(args.eval_steps)
    training_args = TrainingArguments(**training_kwargs)
    collator = Qwen2VlDataCollator(processor=processor, max_length=int(args.max_length))
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_strategy != "no" else None,
        data_collator=collator,
        tokenizer=getattr(processor, "tokenizer", None),
    )


def benchmark_names(raw_value: str) -> List[str]:
    value = str(raw_value or "").strip()
    if not value:
        return ["GQA", "ScienceQA-IMG", "MMBench", "TextVQA", "POPE"]
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def build_eval_dataset_mapping(benchmarks: Sequence[str]) -> Dict[str, str]:
    default_mapping = {
        "GQA": "GQA_TestDev_Balanced",
        "ScienceQA-IMG": "ScienceQA_VAL",
        "MMBench": "MMBench_DEV_EN",
        "TextVQA": "TextVQA_VAL",
        "POPE": "POPE",
    }
    return {name: default_mapping.get(name, name) for name in benchmarks}


def save_adapter_for_eval(trainer: Any, processor: Any, output_dir: Path, args: argparse.Namespace) -> Path:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(adapter_dir))
    try:
        processor.save_pretrained(str(adapter_dir))
    except Exception as exc:
        print(f"[EvalExport] Warning: failed to save processor to adapter dir: {exc}")
    write_json(
        adapter_dir / "adapter_info.json",
        {
            "base_model_path": args.model_name_or_path,
            "finetune_mode": args.finetune_mode,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    return adapter_dir


def maybe_merge_lora_for_eval(model: Any, processor: Any, output_dir: Path, args: argparse.Namespace) -> Tuple[Optional[Path], str]:
    if not bool(getattr(args, "merge_lora_for_eval", False)):
        return None, "disabled"
    if args.finetune_mode not in {"lora", "qlora"}:
        return None, "not_lora_mode"

    merged_dir = output_dir / "merged_model"
    try:
        if not hasattr(model, "merge_and_unload"):
            return None, "model_has_no_merge_and_unload"
        merged_model = model.merge_and_unload()
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        processor.save_pretrained(str(merged_dir))
        return merged_dir, "ok"
    except Exception as exc:
        return None, f"failed: {exc}"


def write_vlmeval_config(
    output_dir: Path,
    args: argparse.Namespace,
    eval_model_path: str,
    benchmarks: Sequence[str],
    dataset_mapping: Dict[str, str],
) -> Path:
    model_key = sanitize_name(args.vlmeval_model_key)
    config = {
        "model": {
            model_key: {
                "class": args.vlmeval_model_class,
                "model_path": eval_model_path,
                "model": eval_model_path,
                "pretrained": eval_model_path,
                "trust_remote_code": bool(args.trust_remote_code),
            }
        },
        "data": {
            dataset_mapping.get(name, name): {}
            for name in benchmarks
        },
    }
    config_path = output_dir / "vlmeval_config.json"
    write_json(config_path, config)
    return config_path


def make_shell_join(items: Sequence[str]) -> str:
    return " ".join(str(item) for item in items if str(item).strip())


def write_evaluation_plan(
    output_dir: Path,
    args: argparse.Namespace,
    adapter_path: Path,
    merged_model_path: Optional[Path],
    merge_status: str,
    best_checkpoint: str,
    metrics: Dict[str, Any],
) -> None:
    benchmarks = benchmark_names(args.eval_benchmarks)
    dataset_mapping = build_eval_dataset_mapping(benchmarks)
    eval_model_path = str(merged_model_path or adapter_path)
    recommended_model_source = "merged_model" if merged_model_path is not None else "adapter"
    vlmeval_output_dir = output_dir / "vlmevalkit_outputs"
    lmms_output_dir = output_dir / "lmms_eval_outputs"
    config_path = write_vlmeval_config(
        output_dir,
        args=args,
        eval_model_path=eval_model_path,
        benchmarks=benchmarks,
        dataset_mapping=dataset_mapping,
    )
    vlmeval_data_names = [dataset_mapping.get(name, name) for name in benchmarks]
    model_key = sanitize_name(args.vlmeval_model_key)

    vlmeval_python_command = (
        f"python $VLMEVALKIT_ROOT/run.py --config {config_path} "
        f"--work-dir {vlmeval_output_dir} --mode all --verbose"
    )
    vlmeval_torchrun_command = (
        f"torchrun --nproc-per-node=$VLMEVAL_NPROC $VLMEVALKIT_ROOT/run.py "
        f"--config {config_path} --work-dir {vlmeval_output_dir} --mode all --verbose"
    )
    vlmeval_model_command = (
        f"python $VLMEVALKIT_ROOT/run.py --data {make_shell_join(vlmeval_data_names)} "
        f"--model {model_key} --work-dir {vlmeval_output_dir} --mode all --verbose"
    )
    lmms_tasks = {
        "GQA": "gqa",
        "ScienceQA-IMG": "scienceqa_img",
        "MMBench": "mmbench_en_dev",
        "TextVQA": "textvqa_val",
        "POPE": "pope",
    }
    lmms_task_names = [lmms_tasks.get(name, name) for name in benchmarks]
    lmms_command = (
        f"lmms_eval --model qwen2_vl --model_args pretrained={eval_model_path} "
        f"--tasks {','.join(lmms_task_names)} --output_path {lmms_output_dir}"
    )

    payload = {
        "status": "ready",
        "eval_backend": "vlmevalkit",
        "fallback_eval_backend": "lmms-eval",
        "message": (
            "Training is handled by this project. Benchmark inference/evaluation is delegated "
            "to VLMEvalKit by default; lmms-eval commands are provided as a fallback."
        ),
        "model_name_or_path": args.model_name_or_path,
        "base_model_path": args.model_name_or_path,
        "adapter_path": str(adapter_path),
        "merged_model_path": None if merged_model_path is None else str(merged_model_path),
        "merge_lora_for_eval": bool(args.merge_lora_for_eval),
        "merge_status": merge_status,
        "recommended_model_source": recommended_model_source,
        "recommended_model_path": eval_model_path,
        "best_checkpoint": best_checkpoint,
        "last_checkpoint": metrics.get("last_checkpoint"),
        "subset_mode": args.subset_mode,
        "subset_ratio": args.subset_ratio,
        "seed": int(args.seed),
        "benchmarks": list(benchmarks),
        "vlmevalkit_dataset_mapping": dataset_mapping,
        "vlmevalkit_model_key": model_key,
        "vlmevalkit_model_class": args.vlmeval_model_class,
        "vlmevalkit_config_path": str(config_path),
        "output_dir": str(output_dir),
        "vlmevalkit_output_dir": str(vlmeval_output_dir),
        "lmms_eval_output_dir": str(lmms_output_dir),
        "recommended_commands": {
            "vlmevalkit_config_python": vlmeval_python_command,
            "vlmevalkit_config_torchrun": vlmeval_torchrun_command,
            "vlmevalkit_supported_model_key": vlmeval_model_command,
            "lmms_eval_fallback": lmms_command,
        },
        "manual_checks": [
            "Confirm VLMEvalKit has a Qwen2-VL model class matching vlmevalkit_model_class, or edit vlmeval_config.json to match your installed VLMEvalKit version.",
            "If evaluating adapter_path directly fails, enable --merge_lora_for_eval and use merged_model_path.",
            "Confirm VLMEvalKit dataset names match your local version; mapping is recorded in vlmevalkit_dataset_mapping.",
        ],
    }
    write_json(output_dir / "benchmark_eval_plan.json", payload)
    command_path = output_dir / "run_vlmevalkit_eval.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        ": \"${VLMEVALKIT_ROOT:?Set VLMEVALKIT_ROOT to your VLMEvalKit checkout}\"\n"
        f"{vlmeval_python_command}\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal Qwen2-VL LoRA/QLoRA finetuning on LLaVA-style instruction subsets.")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="llava_instruct_150k")
    parser.add_argument("--annotation_path", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--val_annotation_path", type=str, default="")
    parser.add_argument("--output_root", type=str, default="artifacts/vlm_finetune")
    parser.add_argument("--subset_mode", type=str, default="full", choices=["full", "random", "ours"])
    parser.add_argument("--subset_ratio", type=float, default=None, help="Subset percentage for random/ours labels, e.g. 1, 5, 10.")
    parser.add_argument("--selected_indices_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.02)

    parser.add_argument("--finetune_mode", type=str, default="lora", choices=["lora", "qlora", "full"])
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--disable_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--device_map", type=str, default="")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=True)
    parser.add_argument("--report_to", type=str, default="none")

    parser.add_argument("--eval_config", type=str, default="")
    parser.add_argument("--eval_benchmarks", type=str, default="GQA,ScienceQA-IMG,MMBench,TextVQA,POPE")
    parser.add_argument("--eval_backend", type=str, default="vlmevalkit", choices=["vlmevalkit", "lmms-eval"])
    parser.add_argument("--vlmeval_model_key", type=str, default="qwen2vl_subset")
    parser.add_argument("--vlmeval_model_class", type=str, default="Qwen2VLChat")
    parser.add_argument("--merge_lora_for_eval", action="store_true", default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    records = load_json_or_jsonl(args.annotation_path)
    selected_records, selected_indices = select_records(
        records,
        subset_mode=args.subset_mode,
        subset_ratio=args.subset_ratio,
        selected_indices_path=args.selected_indices_path or None,
        seed=args.seed,
    )
    val_records = load_json_or_jsonl(args.val_annotation_path) if args.val_annotation_path else None
    train_records, eval_records = split_train_val(selected_records, val_records, args.val_ratio, args.seed)

    output_dir = build_output_dir(args, subset_size=len(selected_records))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "subset_info.json",
        {
            "dataset_name": args.dataset_name,
            "annotation_path": args.annotation_path,
            "image_root": args.image_root,
            "subset_mode": args.subset_mode,
            "subset_ratio": args.subset_ratio,
            "selected_indices_path": args.selected_indices_path,
            "num_total_records": len(records),
            "num_selected_records": len(selected_records),
            "num_train_records": len(train_records),
            "num_eval_records": len(eval_records),
            "seed": int(args.seed),
            "selected_indices_preview": selected_indices[:20],
        },
    )

    train_dataset = LlavaInstructionDataset(train_records, image_root=args.image_root)
    eval_dataset = LlavaInstructionDataset(eval_records, image_root=args.image_root) if eval_records else None

    model, processor = load_model_and_processor(args)
    trainer = build_trainer(args, model, processor, train_dataset, eval_dataset, output_dir)

    start = time.time()
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "last_checkpoint"))
    adapter_path = save_adapter_for_eval(trainer, processor, output_dir, args)
    if getattr(trainer.state, "best_model_checkpoint", None):
        best_checkpoint = trainer.state.best_model_checkpoint
    else:
        best_checkpoint = str(output_dir / "last_checkpoint")

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "elapsed_seconds": float(time.time() - start),
            "best_model_checkpoint": best_checkpoint,
            "last_checkpoint": str(output_dir / "last_checkpoint"),
            "subset_mode": args.subset_mode,
            "subset_ratio": args.subset_ratio,
            "num_selected_records": len(selected_records),
            "num_train_records": len(train_records),
            "num_eval_records": len(eval_records),
            "finetune_mode": args.finetune_mode,
            "model_name_or_path": args.model_name_or_path,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    if eval_dataset is not None and len(eval_dataset) > 0:
        eval_metrics = trainer.evaluate()
        metrics.update({f"final_{key}": value for key, value in eval_metrics.items()})
    write_json(output_dir / "metrics.json", metrics)
    merged_model_path, merge_status = maybe_merge_lora_for_eval(trainer.model, processor, output_dir, args)
    write_evaluation_plan(
        output_dir,
        args,
        adapter_path=adapter_path,
        merged_model_path=merged_model_path,
        merge_status=merge_status,
        best_checkpoint=best_checkpoint,
        metrics=metrics,
    )

    print("VLM finetuning finished:")
    print(f"  output_dir: {output_dir}")
    print(f"  metrics_path: {output_dir / 'metrics.json'}")
    print(f"  best_checkpoint: {best_checkpoint}")
    print(f"  adapter_path: {adapter_path}")
    print(f"  merged_model_path: {merged_model_path}")
    print(f"  benchmark_eval_plan: {output_dir / 'benchmark_eval_plan.json'}")


if __name__ == "__main__":
    main()
