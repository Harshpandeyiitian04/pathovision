# PathoVision

PathoVision is a PyTorch research project for classifying breast histopathology images from the BreakHis dataset as **benign** or **malignant**. It divides each image into tissue-containing tiles, classifies the tiles with a ResNet18 model, and combines their scores into one image-level prediction.

> **Research use only.** PathoVision has not been clinically validated. It must not be used to diagnose disease or guide treatment.

## How it works

1. Read image labels, patient IDs, subtypes, and magnifications from the dataset folders and filenames.
2. Split the data by patient into training, validation, and test sets. Images from one patient stay in one set.
3. Cut images into overlapping 224 x 224 tiles and skip tiles with too little tissue.
4. Train an ImageNet-initialized ResNet18 to estimate the malignant probability of each tile.
5. Combine tile probabilities with mean, max, or top-K aggregation to get an image-level result.
6. Report classification metrics and create plots. Grad-CAM is available for inspecting a tile's activations.

The default settings are in `configs/default.yaml`. The tile size, stride, tissue filter, split proportions, training options, and prediction threshold can be changed there.

## Dataset

BreakHis images are not included as part of the source code. Obtain the dataset under its own terms, then place the extracted files under `data/raw/`. The default parser looks for `benign` and `malignant` in the folder path and reads magnification and patient information from the BreakHis filename or folder structure. If your copy is elsewhere, update `dataset.root` in `configs/default.yaml`.

For example, the configured root should contain the extracted dataset folders somewhere below it:

```text
data/raw/
`-- BreaKHis_v1/
    `-- histology_slides/
        `-- breast/
            |-- benign/
            `-- malignant/
```

## Installation

Use Python 3.10 or later. From the repository directory, create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Install the project requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The generic PyTorch requirement may install a CPU build. For GPU training, install the PyTorch build that matches your CUDA setup, following the instructions for your environment.

The first run with the default `model.pretrained: true` setting needs access to the ImageNet ResNet18 weights. If those weights are unavailable, set `model.pretrained` to `false` in the configuration.

## Run the pipeline

### 1. Prepare and check the dataset

```bash
python scripts/prepare_data.py --config configs/default.yaml --validate-images
```

This creates metadata and patient-level `train.csv`, `val.csv`, and `test.csv` split files. It also reports dataset and split counts. Image validation checks that files can be opened.

### 2. Train the model

```bash
python scripts/train.py --config configs/default.yaml
```

Training saves the best validation checkpoint to `checkpoints/best_model.pth`. The checkpoint is selected using validation ROC-AUC when available, or validation F1 when ROC-AUC cannot be calculated. Test data is not used for checkpoint selection.

For a smaller run, set `training.mode` to `fast` in the configuration. Fast mode applies the image limits and training settings listed in its `fast` section.

### 3. Evaluate on the test split

```bash
python scripts/evaluate.py --config configs/default.yaml
```

Evaluation compares mean, max, and top-K tile aggregation and writes metrics for the configured method. To also create a Markdown summary from the generated results, run:

```bash
python scripts/generate_final_report.py
```

### 4. Predict one image (optional)

After training, run:

```bash
python scripts/predict.py --config configs/default.yaml --image path/to/image.png
```

To create a Grad-CAM overlay for the highest-scoring tile, add `--gradcam`:

```bash
python scripts/predict.py --config configs/default.yaml --image path/to/image.png --gradcam
```

Single-image predictions and visualizations are saved under `results/inference/` by default. Use `--save-dir` to choose another output directory.

## Outputs

The pipeline writes generated files to these locations:

| Location | Contents |
| --- | --- |
| `data/metadata/metadata.csv` | Parsed image metadata |
| `data/splits/` | Patient-level training, validation, and test CSV files |
| `checkpoints/` | Best model checkpoint, its configuration, and training history |
| `results/metrics/` | Test metrics, aggregation comparison, magnification metrics, and report data |
| `results/predictions/` | Image-level and tile-level test predictions |
| `results/figures/` | Dataset, training, ROC, and confusion-matrix plots |
| `results/FINAL_RESULTS.md` | Optional Markdown summary created by the report script |

Results depend on the dataset and training run. No performance numbers are claimed here; use the files generated by your own run.

## Tests

Run the test suite with:

```bash
pytest -q
```

The tests use small synthetic inputs and do not require the full BreakHis dataset.

## Repository layout

```text
configs/       Default experiment settings
notebooks/     Google Colab walkthrough
scripts/       Data preparation, training, evaluation, and prediction commands
src/data/      BreakHis parsing, patient splits, datasets, and tiling
src/models/    ResNet18 classifier
src/training/  Training and validation loops
src/evaluation/ Metrics, aggregation, and test evaluation
src/explainability/  Grad-CAM utilities
tests/         Unit tests
```

The notebook at `notebooks/PathoVision_Colab.ipynb` follows the same workflow for Google Colab. Provide the dataset in the expected location before running its data and training cells.

## Limitations and license

This is an experimental baseline for learning and research. Its results may be affected by the dataset's size, patient mix, staining, image acquisition, and other dataset-specific factors. It has no external hospital validation or clinical validation.

The repository's MIT license applies to its source code. It does not grant rights to the BreakHis dataset; follow the dataset provider's terms when obtaining and using those images.
