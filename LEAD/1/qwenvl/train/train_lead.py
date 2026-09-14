import logging
import os
import pathlib
import sys
from pathlib import Path

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.config import BASE_MODEL_PATH
from attention import qwen3vl_forward, return_mask
import model_lead as model_module
from model_lead import Qwen3VLForConditionalGeneration
from qwenvl.data.dataset import make_supervised_data_module
from qwenvl.train.argument import DataArguments, ModelArguments, TrainingArguments
from lead_trainer import Trainer as Qwen3VLTrainer
from transformers import AutoProcessor
local_rank = None


def configure_attention():
    model_module.Qwen3VLTextAttention.forward = qwen3vl_forward
    model_module.create_causal_mask = return_mask


class LossLoggingTrainer(Qwen3VLTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._extra_loss_sums = {
            "train": {"lm": 0.0, "cls": 0.0, "total": 0.0, "count": 0},
            "eval": {"lm": 0.0, "cls": 0.0, "total": 0.0, "count": 0},
        }

    def _record_extra_losses(self, outputs, phase):
        if not isinstance(outputs, dict):
            return

        lm_loss = outputs.get("lm_loss")
        cls_loss = outputs.get("cls_loss")
        total_loss = outputs.get("loss")
        if lm_loss is None and cls_loss is None:
            return

        bucket = self._extra_loss_sums[phase]
        if lm_loss is not None:
            bucket["lm"] += self._distributed_loss_mean(lm_loss)
        if cls_loss is not None:
            bucket["cls"] += self._distributed_loss_mean(cls_loss)
        if total_loss is not None:
            bucket["total"] += self._distributed_loss_mean(total_loss)
        bucket["count"] += 1

    def _distributed_loss_mean(self, loss):
        loss_tensor = loss.detach().float().mean()
        if hasattr(self, "accelerator") and self.accelerator is not None:
            loss_tensor = self.accelerator.gather_for_metrics(loss_tensor.reshape(1)).mean()
        return loss_tensor.item()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )
        loss, outputs = result
        self._record_extra_losses(outputs, "train" if model.training else "eval")
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            self._inject_extra_loss_logs(logs, "train")
        if "eval_loss" in logs:
            self._inject_extra_loss_logs(logs, "eval")
        return super().log(logs, start_time=start_time)

    def _inject_extra_loss_logs(self, logs, phase):
        bucket = self._extra_loss_sums[phase]
        count = bucket["count"]
        if count == 0:
            return

        prefix = "train" if phase == "train" else "eval"
        logs[f"{prefix}_lm_loss"] = round(bucket["lm"] / count, 6)
        logs[f"{prefix}_cls_loss"] = round(bucket["cls"] / count, 6)
        logs[f"{prefix}_total_loss_logged"] = round(bucket["total"] / count, 6)
        bucket.update({"lm": 0.0, "cls": 0.0, "total": 0.0, "count": 0})


MODULE_TRAINING = {
    "use_lora": True,
    "train_visual": True,
    "train_multi_heads_classifier": True,
    "train_classes_feat_fusion": True,
    "train_injection_blocks": True,
    "train_lm_head": False,
    "train_full_llm": False,
}

INJECTION_LAYER_INDICES = list(range(36))
CLS_LOSS_WEIGHT = 2.0


def rank0_print(*args):
    if local_rank in (None, -1, 0):
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def get_core_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def collect_modules_to_save():
    modules_to_save = [
        "visual",
        "multi_heads_classifier",
        "classes_feat_fusion",
        "injection_blocks",
    ]
    if MODULE_TRAINING["train_lm_head"]:
        modules_to_save.append("lm_head")
    return modules_to_save


def set_module_trainability(module, should_train: bool):
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = should_train


def apply_trainability_selection(model):
    wrapper_model = get_core_model(model)
    base_model = wrapper_model.model

    if MODULE_TRAINING["use_lora"]:
        return

    set_module_trainability(base_model.language_model, MODULE_TRAINING["train_full_llm"])
    if not MODULE_TRAINING["train_full_llm"]:
        set_module_trainability(wrapper_model.lm_head, MODULE_TRAINING["train_lm_head"])

    set_module_trainability(base_model.visual, MODULE_TRAINING["train_visual"])
    set_module_trainability(
        getattr(base_model, "multi_heads_classifier", None),
        MODULE_TRAINING["train_multi_heads_classifier"],
    )
    set_module_trainability(
        getattr(base_model, "classes_feat_fusion", None),
        MODULE_TRAINING["train_classes_feat_fusion"],
    )
    set_module_trainability(
        getattr(base_model.language_model, "injection_blocks", None),
        MODULE_TRAINING["train_injection_blocks"],
    )


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    args_list = [
        "--model_name_or_path", BASE_MODEL_PATH,
        "--tune_mm_llm", "False",
        "--tune_mm_vision", "False",
        "--tune_mm_mlp", "False",
        "--dataset_use", "reports%100",
        "--output_dir", str(project_root / "checkpoints"),
        "--cache_dir", str(project_root / "cache"),
        "--bf16",
        "--per_device_train_batch_size", "5",
        "--per_device_eval_batch_size", "5",
        "--gradient_accumulation_steps", "6",
        "--learning_rate", "1e-4",
        "--mm_projector_lr", "1e-4",
        "--vision_tower_lr", "1e-4",
        "--optim", "adamw_torch",
        "--model_max_length", "4096",
        "--data_flatten", "True",
        "--data_packing", "False",
        "--max_pixels", "235200",
        "--min_pixels", "784",
        "--num_train_epochs", "6",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--weight_decay", "0.01",
        "--logging_steps", "10",
        "--eval_strategy", "steps",
        "--eval_steps", "100",
        "--save_strategy", "best",
        "--load_best_model_at_end", "True",
        "--metric_for_best_model", "eval_lm_loss",
        "--greater_is_better", "False",
        "--save_total_limit", "6",
        "--save_only_model", "True",
        "--lora_enable", str(MODULE_TRAINING["use_lora"]),
        "--lora_r", "64",
        "--lora_alpha", "128",
        "--lora_dropout", "0.05",
    ]

    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=args_list)
    local_rank = training_args.local_rank
    data_args.model_type = "qwen3vl"
    os.makedirs(training_args.output_dir, exist_ok=True)

    rank0_print("Trainable modules:", MODULE_TRAINING)
    rank0_print("Injection layers:", INJECTION_LAYER_INDICES)
    rank0_print("Classification loss weight:", CLS_LOSS_WEIGHT)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    model.config.cls_loss_weight = CLS_LOSS_WEIGHT
    model.config.inject_layer_indices = INJECTION_LAYER_INDICES

    configure_attention()
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=collect_modules_to_save(),
        )
        model = get_peft_model(model, lora_config)

    apply_trainability_selection(model)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = LossLoggingTrainer(model=model, processing_class=processor.tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
