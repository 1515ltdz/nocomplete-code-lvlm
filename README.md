# LEAD v1

## 运行前准备

1. **运行环境**：准备 Linux、Bash、tmux 和支持 BF16 的 NVIDIA GPU。训练启动脚本默认使用两张 GPU（编号 0、1）；推理可使用单卡或多卡。安装相互兼容的 CUDA、PyTorch、Transformers、PEFT、Accelerate、FlashAttention，以及 pandas、NumPy、Pillow、tqdm、safetensors。Transformers 需包含本代码使用的 Qwen3-VL 接口。
2. **基础模型**：准备完整的 Qwen3-VL-8B-Instruct 模型目录，包括权重、模型配置、tokenizer 和 processor 文件。推理还需准备由相同 LEAD 架构训练的 checkpoint，并通过 `--adapter_model_path` 指定；该 checkpoint 应包含 LoRA 和额外训练模块的权重。
3. **数据**：提前完成图像预处理和训练、验证、测试集划分。每条记录包含一张图像的路径和图文对话，格式见下方示例。训练和验证还需要分类标签 CSV，其中 `path_to_image` 必须与 JSON 中的 `image` 值完全一致，标签表须覆盖两个集合。14 类列名见下方；值为 1 表示阳性，当前代码将其他数值和空值视为 0。
4. **路径与输出**：按下表放置文件，或在启动前设置对应环境变量，建议使用绝对路径。图像读取位置为图像根目录与 `image` 相对路径的组合。训练输出目录为项目根目录下的 `checkpoints`；开始独立的新训练前，确保其中没有旧的 `checkpoint-*`，否则入口会尝试恢复训练。为模型和结果预留足够的显存与磁盘空间。

本目录提供训练和 checkpoint 推理代码，不包含数据预处理、数据集、基础模型或训练权重。数据文件需符合下述格式；代码不会自动查找或下载 CheXpertPlus。依赖版本尚未在本目录中锁定。

## Before running

1. **Environment**: Prepare Linux, Bash, tmux, and NVIDIA GPUs with BF16 support. The training launcher uses two GPUs (IDs 0 and 1); inference supports one or multiple GPUs. Install compatible versions of CUDA, PyTorch, Transformers, PEFT, Accelerate, FlashAttention, pandas, NumPy, Pillow, tqdm, and safetensors. Transformers must provide the Qwen3-VL interfaces used by this code.
2. **Base model**: Prepare the complete Qwen3-VL-8B-Instruct directory, including weights, configuration, tokenizer, and processor files. Inference also requires a checkpoint trained with the same LEAD architecture, specified through `--adapter_model_path`. It must contain the LoRA weights and the additional trained modules.
3. **Data**: Complete image preprocessing and the train/validation/test split beforehand. Each record contains one image path and a conversation, as illustrated below. Training and validation also require a label CSV: its `path_to_image` values must exactly match the JSON `image` values, and it must cover both splits. Use the 14 column names listed below. A value of 1 is positive; the current code maps other numeric values and missing values to 0.
4. **Paths and outputs**: Use the default layout below or set the corresponding environment variables before launch. Absolute paths are recommended. Images are located by joining the image root with the relative `image` path. Training writes to `checkpoints` under the project root. For a separate new run, ensure it contains no old `checkpoint-*` directories, as the entry point otherwise attempts to resume training. Allow sufficient GPU memory and disk space for models and outputs.

This directory provides training and checkpoint inference code. Preprocessing tools, datasets, base models, and trained weights are not included. Input files must follow the format below; the code does not automatically locate or download CheXpertPlus. Dependency versions are not pinned in this directory.

## 路径配置 / Path configuration

默认位置相对于项目根目录。单独指定某个文件的环境变量优先于数据根目录的默认设置。

Default locations are relative to the project root. A file-specific environment variable overrides the default derived from the data root.

| 资源 / Resource | 环境变量 / Environment variable | 默认位置 / Default |
| --- | --- | --- |
| 数据根目录 / Data root | `LEAD_DATA_ROOT` | `data` |
| 训练集 / Training split | `LEAD_TRAIN_FILE` | `<data root>/train.json` |
| 验证集 / Validation split | `LEAD_VAL_FILE` | `<data root>/val.json` |
| 测试集 / Test split | `LEAD_TEST_FILE` | `<data root>/test.json` |
| 分类标签 / Classification labels | `LEAD_LABEL_FILE` | `<data root>/labels.csv` |
| 图像根目录 / Image root | `LEAD_IMAGE_ROOT` | `<data root>` |
| 基础模型 / Base model | `LEAD_BASE_MODEL` | `models/base` |

示例：文件按默认名称存放在外部目录时，只需设置以下变量；路径均为占位示例。

Example: when files use the default names in an external directory, set the following variables. These paths are placeholders.

```bash
export LEAD_DATA_ROOT="/path/to/dataset"
export LEAD_IMAGE_ROOT="/path/to/dataset"
export LEAD_BASE_MODEL="/path/to/Qwen3-VL-8B-Instruct"
```

推理时，显式传入的 `--test_file`、`--img_root`、`--base_model_path` 会覆盖上述配置；`--adapter_model_path` 为必填项。

For inference, explicit `--test_file`, `--img_root`, and `--base_model_path` arguments override this configuration. `--adapter_model_path` is required.

## 数据格式 / Data format

每个划分文件是一个 JSON 列表。以下内容仅为格式示例，不是真实病例。

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

训练和验证标签 CSV 必须包含以下表头，其中 `path_to_image` 对应上例的 `images/example.jpg`。

The training/validation label CSV must include these columns. For the example above, `path_to_image` is `images/example.jpg`.

```csv
path_to_image,No Finding,Enlarged Cardiomediastinum,Cardiomegaly,Lung Lesion,Lung Opacity,Edema,Consolidation,Pneumonia,Atelectasis,Pneumothorax,Pleural Effusion,Pleural Other,Fracture,Support Devices
```
