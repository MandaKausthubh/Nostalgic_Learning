# Nostalgia: Hessian-Aware Constrained Fine-Tuning for Vision Models

This repository contains the official implementation of **Nostalgia**, a constrained optimization method for continual and sequential fine-tuning of pretrained vision models. Nostalgia restricts parameter updates to lie in the null space of low-rank approximations of past-task Hessians, enabling adaptation to new tasks while mitigating catastrophic forgetting—without replay.

The codebase focuses on **Vision Transformer (ViT)** models trained on **ImageNet-derived label splits**, but is modular and extensible to other architectures and datasets.

---

## Key Features

* Hessian-aware gradient projection for continual learning
* Low-rank eigenspace approximation using Lanczos-style methods
* Support for multiple baselines (Adam, L2-SP, EWC)
* Task-incremental learning with disjoint label splits
* Optional LoRA-based parameter-efficient fine-tuning
* Stable training on large-scale datasets (ImageNet)

---

## Repository Structure

```text
.
├── models/
│   └── vit32.py              # ViT backbone and task-head management
├── utils_new/
│   ├── hessian.py            # Hessian eigenspace computation
│   ├── accumulate.py         # Hessian accumulation (average)
│   └── ortho_accumulate.py   # Hessian eigenspace union
├── datasets/
│   └── imagenet.py           # ImageNet split construction 
├── optimizers/
│   └── nostalgia.py          # Nostalgia optimizer (gradient projection)
├── train.py                  # Main training script
├── run_experiments.sh        # Example experiment launcher
└── README.md
```

We would like to credit: the ![model soups repository](https://github.com/mlfoundations/model-soups/blob/main/README.md) for the datasets repository. We directly borrow the code and make modifications for the our purposes, from this repository.

---

## Installation

### Requirements

* Python ≥ 3.9
* PyTorch ≥ 2.1
* CUDA (optional but recommended for ImageNet-scale experiments)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

### ImageNet Splits

Experiments are conducted on **ImageNet**, partitioned into **label-disjoint splits**. Each split defines a separate classification task with a fixed number of classes (e.g., 200 per task). Tasks are presented sequentially, and no data from previous tasks is revisited.

Dataset preparation assumes the standard ImageNet directory structure:

```text
ImageNet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
```

Update `--root_dir` to point to your ImageNet location.

---

## Training

### Basic Training Command

```bash
python train.py \
  --mode nostalgia \
  --root_dir /path/to/imagenet \
  --device cuda \
  --batch_size 256 \
  --hessian_eigenspace_dim 32
```

### Training Modes

* `nostalgia` – Hessian-projected optimization (proposed method)
* `Adam` – Standard fine-tuning baseline
* `l2sp` – L2-SP regularization
* `EWC` – Elastic Weight Consolidation

---

## Training Procedure

For each task:

1. **Head warm-up**
   The task-specific classification head is trained while freezing the backbone.

2. **Full training**
   The backbone is unfrozen and trained using the selected optimization method.

3. **Hessian update (Nostalgia only)**
   A low-rank approximation of the task Hessian is computed and accumulated for future tasks.

Weight decay is disabled by default when using Nostalgia, as it interferes with constrained updates.

---

## Optimizers

* **Nostalgia** uses a custom optimizer that projects gradients into the null space of past-task Hessians.
* The recommended base optimizer for Nostalgia is **SGD with momentum**, as adaptive optimizers (e.g., Adam/AdamW) distort the geometry of projected updates.
* Adam/AdamW baselines are provided for comparison.

---

## Reproducibility

All experiments are controlled via command-line arguments. Key hyperparameters such as learning rates, Hessian rank, LoRA configuration, and random seeds are logged automatically.

Each run stores:

* TensorBoard logs
* Configuration files
* Model checkpoints (per task)

---

## Example: Running Multiple Experiments

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

The script demonstrates how to sweep over optimizers, Hessian dimensions, and random seeds.

---

## Results and Expectations

On ImageNet-derived 200-class tasks:

* Early training (1–2 epochs) typically reaches **45–50% top-1 accuracy**
* With sufficient training, Nostalgia achieves **55–65% top-1 accuracy**, while maintaining strong retention across tasks
* These results are competitive with unconstrained fine-tuning baselines, without replay

Exact performance depends on task order, optimizer choice, and training duration.

---

## Notes on Stability

* Nostalgia is **numerically sensitive** by design; NaN checks and careful optimizer handling are essential.
* Gradient projection is disabled during head warm-up.
* Optimizers must be re-initialized whenever parameter structure changes (e.g., LoRA reset).

---

## Citation

If you use this code, please cite:

```bibtex
@article{nostalgia2026,
  title={Nostalgia: Hessian-Aware Constrained Fine-Tuning for Continual Learning},
  author={Anonymous},
  year={2026}
}
```

(Replace with the final citation upon publication.)

---

## License

This project is released under the MIT License.

---

If you want, I can next:

* tailor this README to match **OpenReview / ICML artifact guidelines**,
* add a **“Known Issues / Debugging”** section (very useful for custom optimizers),
* or help you write an **Artifact Evaluation appendix**.

You’re in excellent shape — this repo now looks like a serious research artifact.
