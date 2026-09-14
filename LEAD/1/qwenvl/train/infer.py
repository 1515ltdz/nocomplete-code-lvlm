"Batch inference for a LEAD checkpoint. Each process loads the model on one GPU and processes a dataset shard."
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer


project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.config import BASE_MODEL_PATH, IMAGE_ROOT, TEST_FILE
from model_lead import Qwen3VLForConditionalGeneration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


CLASS_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


def load_tokenizer_and_processor(base_model_path: str, adapter_model_path: str):
    tokenizer_source = adapter_model_path if Path(adapter_model_path).exists() else base_model_path
    processor_source = tokenizer_source

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
    except Exception:
        logger.warning("Failed to load tokenizer from %s; falling back to %s", tokenizer_source, base_model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        processor_source = base_model_path

    try:
        processor = AutoProcessor.from_pretrained(processor_source)
    except Exception:
        logger.warning("Failed to load processor from %s; falling back to %s", processor_source, base_model_path)
        processor = AutoProcessor.from_pretrained(base_model_path)

    processor.tokenizer = tokenizer
    processor.tokenizer.padding_side = "left"
    return tokenizer, processor


def load_model_and_processor(
    base_model_path: str,
    adapter_model_path: str,
    gpu_id: int,
    attn_implementation: str = "flash_attention_2",
):
    device_map = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    logger.info("Loading base model from %s on %s", base_model_path, device_map)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )

    logger.info("Loading LoRA/modules_to_save adapter from %s", adapter_model_path)
    model = PeftModel.from_pretrained(
        model,
        adapter_model_path,
        device_map=device_map,
    )
    model.eval()

    _, processor = load_tokenizer_and_processor(
        base_model_path=base_model_path,
        adapter_model_path=adapter_model_path,
    )
    return model, processor


def load_test_data(test_file: str) -> List[Dict]:
    logger.info("Loading test data from %s", test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_data = []
    for item in data:
        question = ""
        reference_answer = ""
        if "conversations" in item:
            for conv in item["conversations"]:
                if conv["from"] == "human":
                    question = conv["value"].replace("<image>\n", "")
                elif conv["from"] == "gpt":
                    reference_answer = conv["value"]

        test_data.append(
            {
                "image": item.get("image", ""),
                "question": question,
                "reference_answer": reference_answer,
            }
        )
    logger.info("Loaded %s test samples", len(test_data))
    return test_data


def shard_test_data(test_data: List[Dict], shard_id: int, num_shards: int) -> List[Dict]:
    shard = test_data[shard_id::num_shards]
    logger.info(
        "Shard %s/%s contains %s samples",
        shard_id,
        num_shards,
        len(shard),
    )
    return shard


def prepare_batch_inputs(
    processor,
    test_data: List[Dict],
    img_root: Optional[str] = None,
    max_pixels: int = 235200,
    min_pixels: int = 784,
):
    if hasattr(processor, "image_processor"):
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels

    messages_list = []
    for item in test_data:
        if item["image"]:
            item["image_path"] = os.path.join(img_root, item["image"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image_path"]},
                        {"type": "text", "text": item["question"]},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": item["question"]}],
                }
            ]
        messages_list.append(messages)

    input_ids_list = []
    attention_mask_list = []
    pixel_values_list = []
    image_grid_thw_list = []
    sequence_lengths = []
    for single_messages in messages_list:
        single_input = processor.apply_chat_template(
            [single_messages],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=False,
        )

        input_ids_list.append(single_input["input_ids"].squeeze(0))
        attention_mask_list.append(single_input["attention_mask"].squeeze(0))
        sequence_lengths.append(single_input["input_ids"].size(1))

        pv = single_input["pixel_values"]
        pixel_values_list.append(pv.squeeze(0) if pv.dim() == 5 else pv)
        image_grid_thw_list.append(single_input["image_grid_thw"])

    padded_inputs = processor.tokenizer.pad(
        {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
        },
        padding=True,
        return_tensors="pt",
    )

    inputs = {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "pixel_values": torch.cat(pixel_values_list, dim=0),
        "image_grid_thw": torch.cat(image_grid_thw_list, dim=0),
    }
    img_data_dict = {
        "sequence_lengths": sequence_lengths,
        "img_class_labels": None,
    }
    return inputs, img_data_dict


def batch_inference(
    model,
    processor,
    test_data: List[Dict],
    output_csv: str,
    output_cls_csv: str,
    img_root: Optional[str] = None,
    batch_size: int = 32,
    min_new_tokens: int = 50,
    max_new_tokens: int = 100,
    repetition_penalty: float = 1.05,
    length_penalty: float = 1.0,
    num_beams: int = 2,
    max_report_tokens: int = 100,
    max_pixels: int = 235200,
    min_pixels: int = 784,
):
    results = []
    cls_results = []
    hypo_dict = {}
    ref_dict = {}

    logger.info("Starting batch inference with batch_size=%s", batch_size)
    pbar = tqdm(total=len(test_data), desc="Processing")

    for i in range(0, len(test_data), batch_size):
        batch = test_data[i : i + batch_size]

        inputs, img_data_dict = prepare_batch_inputs(
            processor=processor,
            test_data=batch,
            img_root=img_root,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        cls_img_data_dict = {
            "sequence_lengths": list(img_data_dict["sequence_lengths"]),
            "img_class_labels": None,
        }

        cls_logits = None
        cls_probs = None
        cls_preds = None
        cls_source_model = model.get_base_model() if hasattr(model, "get_base_model") else model
        with torch.no_grad():
            _, cls_logits_raw = cls_source_model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                sequence_lengths=cls_img_data_dict["sequence_lengths"],
                img_class_labels=None,
                use_cache=False,
            )

        if cls_logits_raw is not None:
            cls_logits = cls_logits_raw.squeeze(-1).detach().float().cpu()
            cls_probs = torch.sigmoid(cls_logits)
            cls_preds = (cls_probs >= 0.5).to(torch.int64)

        if num_beams > 1:
            expanded_lengths = []
            for length in img_data_dict["sequence_lengths"]:
                expanded_lengths.extend([length] * num_beams)
            img_data_dict["sequence_lengths"] = expanded_lengths

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                img_data_dict=img_data_dict,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_beams=num_beams,
                do_sample=False,
            )

        input_ids = inputs["input_ids"]
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for item, generated_text in zip(batch, output_text):
            token_ids = processor.tokenizer.encode(
                item["reference_answer"],
                add_special_tokens=False,
            )
            if len(token_ids) > max_report_tokens:
                token_ids = token_ids[:max_report_tokens]
            truncated_ref = processor.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            ).strip()

            generated_text = generated_text.strip()
            sample_id = item["image"]
            results.append(
                {
                    "image": sample_id,
                    "question": item["question"],
                    "generated_answer": generated_text,
                    "reference_answer": truncated_ref,
                }
            )
            hypo_dict[sample_id] = [generated_text]
            ref_dict[sample_id] = [truncated_ref]

        if cls_logits is not None and cls_probs is not None and cls_preds is not None:
            for item, logits_row, probs_row, preds_row in zip(batch, cls_logits, cls_probs, cls_preds):
                cls_record = {
                    "image": item["image"],
                    "question": item["question"],
                }
                for label, logit_value, prob_value, pred_value in zip(
                    CLASS_LABELS, logits_row.tolist(), probs_row.tolist(), preds_row.tolist()
                ):
                    cls_record[f"{label}_logit"] = logit_value
                    cls_record[f"{label}_prob"] = prob_value
                    cls_record[f"{label}_pred"] = int(pred_value)
                cls_results.append(cls_record)

        pd.DataFrame(results).to_csv(output_csv, index=False)
        pd.DataFrame(cls_results).to_csv(output_cls_csv, index=False)
        pbar.update(len(batch))

    pbar.close()
    return results, hypo_dict, ref_dict, cls_results


def main():
    parser = argparse.ArgumentParser(
        description="LEAD checkpoint inference"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=BASE_MODEL_PATH,
    )
    parser.add_argument(
        "--adapter_model_path",
        type=str,
        required=True,
        help="Path to the LEAD checkpoint.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=TEST_FILE,
    )
    parser.add_argument(
        "--img_root",
        type=str,
        default=IMAGE_ROOT,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
    )
    parser.add_argument("--output_csv", type=str, default="test_results_beam.csv")
    parser.add_argument("--output_cls_csv", type=str, default="test_result_cls.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_pixels", type=int, default=235200)
    parser.add_argument("--min_pixels", type=int, default=784)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--max_report_tokens", type=int, default=100)
    args = parser.parse_args()

    output_dir = os.path.join(
        args.adapter_model_path,
        args.output_dir,
        f"shard_{args.shard_id}_of_{args.num_shards}",
    )
    os.makedirs(output_dir, exist_ok=True)
    output_csv_path = os.path.join(output_dir, args.output_csv)
    output_cls_csv_path = os.path.join(output_dir, args.output_cls_csv)
    output_result_json = os.path.join(output_dir, "test_result.json")
    output_refs_json = os.path.join(output_dir, "test_refs.json")

    model, processor = load_model_and_processor(
        base_model_path=args.base_model_path,
        adapter_model_path=args.adapter_model_path,
        gpu_id=args.gpu_id,
    )

    test_data = load_test_data(args.test_file)
    test_data = shard_test_data(test_data, args.shard_id, args.num_shards)

    results, hypo_dict, ref_dict, cls_results = batch_inference(
        model=model,
        processor=processor,
        test_data=test_data,
        output_csv=output_csv_path,
        output_cls_csv=output_cls_csv_path,
        img_root=args.img_root,
        batch_size=args.batch_size,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        num_beams=args.num_beams,
        max_report_tokens=args.max_report_tokens,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )

    with open(output_result_json, "w", encoding="utf-8") as f:
        json.dump(hypo_dict, f, ensure_ascii=False, indent=4)
    with open(output_refs_json, "w", encoding="utf-8") as f:
        json.dump(ref_dict, f, ensure_ascii=False, indent=4)

    logger.info("Inference completed on shard %s/%s", args.shard_id, args.num_shards)
    logger.info("Total samples on this shard: %s", len(results))
    logger.info("Classification rows saved on this shard: %s", len(cls_results))


if __name__ == "__main__":
    main()
