import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge LEAD inference results."
    )
    parser.add_argument(
        "--adapter_model_path",
        type=str,
        required=True,
        help="Path to the LEAD checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Shard output root under adapter_model_path.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="test_results_beam.csv",
    )
    parser.add_argument(
        "--output_cls_csv",
        type=str,
        default="test_result_cls.csv",
    )
    parser.add_argument(
        "--output_result_json",
        type=str,
        default="test_result.json",
    )
    parser.add_argument(
        "--output_refs_json",
        type=str,
        default="test_refs.json",
    )
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    root = Path(args.adapter_model_path) / args.output_dir
    shard_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("shard_")])

    if not shard_dirs:
        raise FileNotFoundError(f"No shard directories found under: {root}")

    all_frames = []
    all_cls_frames = []
    merged_result = {}
    merged_refs = {}

    for shard_dir in shard_dirs:
        csv_path = shard_dir / "test_results_beam.csv"
        cls_csv_path = shard_dir / args.output_cls_csv
        result_json_path = shard_dir / "test_result.json"
        refs_json_path = shard_dir / "test_refs.json"

        if csv_path.exists():
            all_frames.append(pd.read_csv(csv_path))
        else:
            print(f"[warn] missing csv: {csv_path}")

        if cls_csv_path.exists():
            all_cls_frames.append(pd.read_csv(cls_csv_path))
        else:
            print(f"[warn] missing cls csv: {cls_csv_path}")

        if result_json_path.exists():
            merged_result.update(load_json(result_json_path))
        else:
            print(f"[warn] missing result json: {result_json_path}")

        if refs_json_path.exists():
            merged_refs.update(load_json(refs_json_path))
        else:
            print(f"[warn] missing refs json: {refs_json_path}")

    merged_csv_path = root / args.output_csv
    merged_cls_csv_path = root / args.output_cls_csv
    merged_result_json_path = root / args.output_result_json
    merged_refs_json_path = root / args.output_refs_json

    if all_frames:
        merged_df = pd.concat(all_frames, ignore_index=True)
        if "image" in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=["image"], keep="first")
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved to: {merged_csv_path}")
        print(f"Merged CSV rows: {len(merged_df)}")

    if all_cls_frames:
        merged_cls_df = pd.concat(all_cls_frames, ignore_index=True)
        if "image" in merged_cls_df.columns:
            merged_cls_df = merged_cls_df.drop_duplicates(subset=["image"], keep="first")
        merged_cls_df.to_csv(merged_cls_csv_path, index=False)
        print(f"Merged classification CSV saved to: {merged_cls_csv_path}")
        print(f"Merged classification CSV rows: {len(merged_cls_df)}")

    with merged_result_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=4)
    print(f"Merged predictions JSON saved to: {merged_result_json_path}")
    print(f"Merged predictions count: {len(merged_result)}")

    with merged_refs_json_path.open("w", encoding="utf-8") as f:
        json.dump(merged_refs, f, ensure_ascii=False, indent=4)
    print(f"Merged references JSON saved to: {merged_refs_json_path}")
    print(f"Merged references count: {len(merged_refs)}")


if __name__ == "__main__":
    main()
