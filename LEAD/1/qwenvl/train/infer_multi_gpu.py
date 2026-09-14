import argparse
import os
import subprocess
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from qwenvl.config import BASE_MODEL_PATH, IMAGE_ROOT, TEST_FILE


SCRIPT_PATH = Path(__file__).with_name(
    "infer.py"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch multi-GPU LEAD inference."
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1",
        help="Comma-separated visible GPU ids, for example: 0,1,2,3",
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
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=235200)
    parser.add_argument("--min_pixels", type=int, default=784)

    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=2)
    parser.add_argument("--max_report_tokens", type=int, default=100)
    return parser.parse_args()


def build_worker_cmd(args, gpu_id: str, shard_id: int, num_shards: int):
    return [
        sys.executable,
        str(SCRIPT_PATH),
        "--gpu_id",
        gpu_id,
        "--shard_id",
        str(shard_id),
        "--num_shards",
        str(num_shards),
        "--base_model_path",
        args.base_model_path,
        "--adapter_model_path",
        args.adapter_model_path,
        "--test_file",
        args.test_file,
        "--img_root",
        args.img_root,
        "--output_dir",
        args.output_dir,
        "--output_csv",
        args.output_csv,
        "--batch_size",
        str(args.batch_size),
        "--max_pixels",
        str(args.max_pixels),
        "--min_pixels",
        str(args.min_pixels),
        "--min_new_tokens",
        str(args.min_new_tokens),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--repetition_penalty",
        str(args.repetition_penalty),
        "--length_penalty",
        str(args.length_penalty),
        "--num_beams",
        str(args.num_beams),
        "--max_report_tokens",
        str(args.max_report_tokens),
    ]


def main():
    args = parse_args()
    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    if not gpu_ids:
        raise ValueError("No gpu ids provided.")

    print(f"Launching {len(gpu_ids)} worker processes with GPUs: {gpu_ids}")
    print(f"Worker script: {SCRIPT_PATH}")

    processes = []
    exit_codes = []
    try:
        for shard_id, gpu_id in enumerate(gpu_ids):
            cmd = build_worker_cmd(args, gpu_id=gpu_id, shard_id=shard_id, num_shards=len(gpu_ids))
            env = os.environ.copy()
            print(f"[launcher] starting shard {shard_id} on gpu {gpu_id}")
            processes.append(subprocess.Popen(cmd, env=env))

        exit_codes = [proc.wait() for proc in processes]
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()

    failed = [code for code in exit_codes if code != 0]
    if failed:
        raise SystemExit(f"Inference workers failed with exit codes: {exit_codes}")

    print("All inference workers completed successfully.")
    print("Each shard result is saved under the adapter checkpoint output subdirectory.")


if __name__ == "__main__":
    main()
