# INR-benchmark

Unified benchmark for implicit-neural, grid, and hybrid representations on synthetic & real signals — code for the preprint **“Grids Often Outperform Implicit Neural Representations.”**
---

## Prerequisites

Before proceeding with installation, ensure the following system requirements are met:

### System Requirements

- **NVIDIA GPU** with compute capability ≥ 6.1 (e.g. RTX 20xx, 30xx, 40xx, A100, etc.)
- **CUDA Toolkit** version **≥ 11.4**, tested with **CUDA 12.1–12.2**
- **NVIDIA Driver** compatible with your CUDA version
- **Linux or WSL2 environment** (tested on Ubuntu 20.04+)
- **Conda** (Anaconda or Miniconda)

### Python/Library Compatibility

| Component     | Recommended Version |
|---------------|---------------------|
| Python        | 3.10                |
| PyTorch       | 2.1.x or 2.2.x      |
| CUDA Toolkit  | 12.1 or 12.2        |
| gcc / g++     | 9–11 (for tiny-cuda-nn) |

---
## Additional Required Dependencies

The following external libraries must be installed manually after setting up the Conda environment:

### 1. tiny-cuda-nn
```
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
### 2. Gaussian Splatting (gsplat)
```
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
pip install -e .
cd ..
```
These packages require a valid CUDA setup and may take several minutes to compile on the first install.

After installing gsplat, you must manually move the INR runner script to its expected location:
```
mv models/run_gsplat.py gsplat/examples/run_gsplat.py
```

This ensures that the gsplat runner is correctly placed within the Gaussian Splatting package.

---

## Repository Structure

```
.
├── band_limit_figs/             # Visualizations of reconstruction vs. signal bandwidth
├── gsplat/                      # Gaussian Splatting implementation
├── models/                      # INR, hybrid, and grid model implementations
├── target_signals/             # Signal generation code and datasets
│   ├── bandlimited_signal.py   # Signal class: BandlimitedSignal
│   ├── eval_DIV2K.py           # Evaluation on DIV2K
│   ├── evaluate_voxel.py       # Evaluation on 3D voxel data
│   ├── gen_heatmap_eval_metric_synthetic.py  # Metric visualization
│   └── utils.py
├── *.sh                        # Shell scripts to train each model on a target signal
├── run_ct.py                  # Script for CT reconstruction
├── run.py                     # Generalized runner for synthetic signals
└── hyperparameters.json       # Central config file for training settings
```

---

## Supported Datasets & Signals

### Synthetic Signals (in `target_signals/`)

Each signal is \~1M values (1000×1000 or 100×100×100), enabling fair compression analysis.

| Signal Type        | Class Name           | Dim   | Description                                    |
| ------------------ | -------------------- | ----- | ---------------------------------------------- |
| Spheres            | `SparseSphereSignal` | 2D/3D | Random circles or spheres with varying scale   |
| Bandlimited        | `BandlimitedSignal`  | 2D/3D | Low-pass filtered noise with varying frequency |
| Sierpinski Triangle| `Sierpinski`         | 2D    | Triangle shaped fractal structure              |
| Star Target        | `StarTarget`         | 2D    | Radial wedges with increasing bandwidth        |

### Real-World Signals

| Dataset       | Class Name      | Description                                                                          |
| ------------- | --------------- | ------------------------------------------------------------------------------------ |
| **DIV2K**     | `RealImage`     | 10 high-res images from DIV2K for image fitting, 4× SR, and denoising                |
| **3D Dragon** | `Voxel_Fitting` | Stanford Dragon. `sparse=True` loads surface version; `False` loads occupancy volume |
| **CT Scan**   | -               | 2D human chest CT slice for reconstruction under sparsity                            |

---

## How to Train

All training scripts use `.sh` or `.py` wrappers and can be launched directly.

### Synthetic Signals

Run individual scripts:

```bash
bash run_2dsierpinski.sh      # Train on 2D Sierpinski
bash run_3dbandlimited.sh     # Train on 3D Bandlimited noise
```

### DIV2K Image Training

```bash
bash run_DIV2K.sh
```

### Stanford Dragon

```bash
bash run_dragon.sh
```

### CT Reconstruction

```bash
python run_ct.py
```

---

## Evaluation Metrics

All reconstructions are evaluated using:

* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index, 2D only)
* **LPIPS** (Learned Perceptual Image Patch Similarity, 2D only)
* **IoU** (Intersection over Union for 3D volumes)

Evaluation heatmaps and visualizations are generated via:

```bash
python target_signals/gen_heatmap_eval_metric_synthetic.py
```

---


## Notes

* All experiments are reproducible with the provided scripts.
* Models are tuned on the Star Target dataset (`target_signals/star_resolution_target_40_1000.npy`).
* Signal generation and preprocessing utilities are located in `target_signals/`.
