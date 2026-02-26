# DR. Nav: Semantic-Geometric Representations for Proactive Dead-End Recovery and Navigation

This repository contains code to train and run a multimodal (camera + LiDAR) dead-end detection model. The main training / data-loading / evaluation logic is under `Scripts/model/` and dataset utilities / bag processing are under `Scripts/data/`.

## Repository layout

- `Scripts/model/`
  - `data_loader.py` - Dataset class (`DeadEndDataset`), training loop (`train_model`), evaluation (`evaluate_model`) and visualization (`visualize_test_results`).
  - `model_CA.py` - Model definition (expected `DeadEndDetectionModel` class).
  - `cross_multi.py`, `model_accuracy.py` - auxiliary model code / metrics.
- `Scripts/data/`
  - `rosbag_processor.py`, `annotation.py`, `visualize_predictions.py` - dataset creation and helpers.

## Quick start

Prerequisites (tested on Linux): Python 3.8+ and a CUDA-enabled machine for GPU training.

Create and activate a virtual environment, then install the typical dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torcheval numpy pillow matplotlib scikit-learn wandb
```

If you use a GPU, install the appropriate `torch` wheel for your CUDA version from https://pytorch.org/.

Note: There is no repository `requirements.txt` in this project; the list above covers the packages the code references. Add any additional packages you need.

## Data format and expected layout

The dataset is expected to be organized into bag folders under a `data_root` path. Each bag folder should contain at least:

- `images/sample_id_*/{front.jpg,side_left.jpg,side_right.jpg}`
- `lidar/sample_id_*/{front.bin,side_left.bin,side_right.bin}` (binary float32 point clouds with 4 values per point: x,y,z,intensity)
- Optional `annotations.json` in the bag root (a dict mapping `sample_id` -> annotation dict with keys such as `is_dead_end`, `front_open`, `side_left_open`, `side_right_open`, and direction vectors).

If no `annotations.json` files are present anywhere in the `data_root`, the code will run in inference mode (no ground-truth evaluation).

## Running training

There are two ways to run training:

1) Use the helper functions from `Scripts/model/data_loader.py` in a small runner script (recommended). Example `train_runner.py`:

```python
from Scripts.model.data_loader import get_memory_efficient_data_loaders, train_model
from Scripts.model.model_CA import DeadEndDetectionModel

data_root = '/path/to/train_bags'
save_dir = '/path/to/saved_models'

train_loader, val_loader = get_memory_efficient_data_loaders(data_root, batch_size=4)
model = DeadEndDetectionModel()
model = train_model(model, train_loader, val_loader, num_epochs=20, device='cuda', save_dir=save_dir)
```

Run it with:

```bash
python train_runner.py
```

2) Use the `__main__` section inside `data_loader.py` (less flexible). Edit the hard-coded `inference` flag and paths at the bottom of the file to toggle training vs inference. Not recommended for repeated experiments.

Training configuration notes
- `train_model` accepts `num_epochs`, `lr`, and `device` (either `'cuda'` or `'cpu'` / a `torch.device`).
- `get_memory_efficient_data_loaders` creates `train_loader` and `val_loader`. Tune `batch_size` and `num_workers` according to your machine.
- The training loop uses a conservative optimizer/scheduler and mixed precision when `device=='cuda'`.

## Running evaluation / visualization (inference)

To run visualization on new data (inference mode) use the `visualize_test_results` function from `data_loader.py`.

Example `inference_runner.py`:

```python
from Scripts.model.data_loader import visualize_test_results

visualize_test_results(
    model_path='/path/to/saved_models/model_best.pth',
    data_root='/path/to/test_bags',
    batch_size=8,
    num_samples=50,
    output_dir='/path/to/output_visualizations'
)
```

```bash
python inference_runner.py
```

The function will attempt to import `DeadEndDetectionModel` from `Scripts/model/model_CA.py` if needed. It saves per-sample images and a summary `predictions.json` in `output_dir`.

## Evaluate with ground truth

If your `data_root` contains `annotations.json` files the `visualize_test_results` function will detect that and run evaluation mode (it will compute accuracy and optionally a confusion matrix if scikit-learn is installed).

For a programmatic evaluation (no visualization) you can call `evaluate_model(model, dataloader, device)` — this returns a dict with `val_loss`, `path_f1` and `dead_end_f1`.

## Common runtime issues and tips

- File not found: Many functions assume image and `.bin` files exist. Make sure your bag folder structure matches the expected layout above. Add explicit checks if your dataset differs.
- Serialization bug: When saving `results` the code builds JSON-ready summaries — if you run into issues, inspect the `results` dict (it should map sample IDs to per-sample dicts).
- CUDA OOM: If you hit out-of-memory errors, reduce `batch_size`, reduce image size in transforms, or train on CPU.
- Mixed precision: The training loop uses AMP by default on CUDA. If you encounter numerical issues, try disabling AMP by forcing `device='cpu'` for debug runs.

## Logging experiment runs

The code attempts to use Weights & Biases (`wandb`). You can disable or remove `wandb` usage if you prefer. If using `wandb`, login beforehand:

```bash
wandb login
```

## Troubleshooting checklist

1. Confirm dataset layout (images, lidar bins, optional annotations).
2. Verify `DeadEndDetectionModel` is importable from `Scripts/model/model_CA.py`.
3. Use small batch sizes for initial runs to verify training/inference works.
4. Watch for silent except blocks — enable printing of caught exceptions for debugging.

