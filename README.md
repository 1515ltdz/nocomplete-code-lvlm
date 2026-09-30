# LEAD v1

## Before running

1. **Environment**: Prepare Linux, Bash, tmux, and NVIDIA GPUs with BF16 support. The training launcher uses two GPUs (IDs 0 and 1); inference supports one or multiple GPUs. Install compatible versions of CUDA, PyTorch, Transformers, PEFT, Accelerate, FlashAttention, pandas, NumPy, Pillow, tqdm, and safetensors. Transformers must provide the Qwen3-VL interfaces used by this code.
2. **Base model**: Prepare the complete Qwen3-VL-8B-Instruct directory, including weights, configuration, tokenizer, and processor files. Inference also requires a checkpoint trained with the same LEAD architecture, specified through `--adapter_model_path`. It must contain the LoRA weights and the additional trained modules.
3. **Data**: Complete image preprocessing and the train/validation/test split beforehand. Each record contains one image path and a conversation, as illustrated below. Training and validation also require a label CSV: its `path_to_image` values must exactly match the JSON `image` values, and it must cover both splits. Use the 14 column names listed below. A value of 1 is positive; the current code maps other numeric values and missing values to 0.
4. **Paths and outputs**: Use the default layout below or set the corresponding environment variables before launch. Absolute paths are recommended. Images are located by joining the image root with the relative `image` path. Training writes to `checkpoints` under the project root. For a separate new run, ensure it contains no old `checkpoint-*` directories, as the entry point otherwise attempts to resume training. Allow sufficient GPU memory and disk space for models and outputs.

This directory provides training and checkpoint inference code. Preprocessing tools, datasets, base models, and trained weights are not included. Input files must follow the format below; the code does not automatically locate or download CheXpertPlus. Dependency versions are not pinned in this directory.

## Path configuration

Default locations are relative to the project root. A file-specific environment variable overrides the default derived from the data root.

| Resource | Environment variable | Default |
| --- | --- | --- |
| Data root | `LEAD_DATA_ROOT` | `data` |
| Training split | `LEAD_TRAIN_FILE` | `<data root>/train.json` |
| Validation split | `LEAD_VAL_FILE` | `<data root>/val.json` |
| Test split | `LEAD_TEST_FILE` | `<data root>/test.json` |
| Classification labels | `LEAD_LABEL_FILE` | `<data root>/labels.csv` |
| Image root | `LEAD_IMAGE_ROOT` | `<data root>` |
| Base model | `LEAD_BASE_MODEL` | `models/base` |

Example: when files use the default names in an external directory, set the following variables. These paths are placeholders.

```bash
export LEAD_DATA_ROOT="/path/to/dataset"
export LEAD_IMAGE_ROOT="/path/to/dataset"
export LEAD_BASE_MODEL="/path/to/Qwen3-VL-8B-Instruct"
```

For inference, explicit `--test_file`, `--img_root`, and `--base_model_path` arguments override this configuration. `--adapter_model_path` is required.

## Data format

Each split is a JSON list. The entry below illustrates the format and is not a real case.

```json
[
  {
    "image": "images/example.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWrite the radiology findings."},
      {"from": "gpt", "value": "Example reference findings."}
    ]
  }
]
```

The training/validation label CSV must include these columns. For the example above, `path_to_image` is `images/example.jpg`.

```csv
path_to_image,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Lesion,Lung Opacity,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
```
