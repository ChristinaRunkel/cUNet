# Continuous U-Net: Faster, Greater, Noiseless

## Overview
This is the offical implementation of the "Continuous U-Net: Faster, Greater, Noiseless" paper (https://openreview.net/pdf?id=ongi2oe3Fr). Continuous U-Net is a continuous deep network whose dynamics are modelled by second order ordinary differential
equations. We view the dynamics in our network as the boxes consisting of CNNs and transform them into dynamic blocks to get a solution. We introduce the first U-Net variant working explicitly in higher-order neural ODEs.

## Features
- 2D segmentation model training and evaluation
- Custom data augmentations and transforms
- Metrics computation for segmentation quality and robustness
- Utilities for mean/std computation, data loading, and experiment management

## Project Structure
```
cUNet/
├── ablation_studies.py           # Ablation study experiments
├── augmentations.py              # Custom data augmentations
├── building_blocks.py            # Model building blocks
├── compute_mean_std.py           # Compute dataset mean/std
├── compute_metrics.py            # Segmentation metrics
├── compute_metrics_block_type.py # Block type metrics
├── compute_metrics_noisy.py      # Metrics under noise
├── cunet.yml                     # CUNet configuration
├── data.py                       # Data loading utilities
├── metrics.py                    # Metric implementations
├── models.py                     # Model architectures
├── other_models.py               # Additional models
├── script.py                     # Main training script
├── train_utils.py                # Training utilities
├── transforms.py                 # Data transforms
```

## Installation
1. Clone the repository:
   ```bash
   git clone <REPO_URL>
   cd cUNet
   ```
2. Install dependencies (Python 3.8+ recommended):
   ```bash
  conda env create -f cunet.yml
  conda activate cunet
   ```
   Or install main libraries manually:
   - torch, torchvision
   - numpy, scipy, scikit-image
   - monai, segmentation_models_pytorch
   - ml_collections

## Usage
- **Training:**
  ```bash
  python script.py --config <CONFIG_FILE>
  ```
- **Ablation Studies:**
  ```bash
  python ablation_studies.py --config <CONFIG_FILE>
  ```
- **Metrics Computation:**
  ```bash
  python compute_metrics.py --input <PREDICTIONS> --labels <LABELS>
  ```

## Configuration
- Update paths in scripts using placeholders (e.g., `<PATH_TO_DATASET>`, `<PATH_TO_OUTPUTS_DIR>`).

## Data
- The code expects medical imaging data in NIfTI format (`.nii`/`.nii.gz`).
- Update data paths in scripts or configs as needed.

## Citation
If you use this code in your research, please cite:
```
@article{cheng2024continuous,
  title={Continuous {U-Net}: Faster, Greater and Noiseless},
  author={Cheng, Chun-Wun and Runkel, Christina and Liu, Lihao and Chan, Raymond H and Sch{\"o}nlieb, Carola-Bibiane and Aviles-Rivero, Angelica I},
  journal={Transactions on Machine Learning Research},
  year={2024}
}
```

## License
This project is licensed under the MIT License.

## Contact
For questions or contributions, please contact: cwc56@cam.ac.uk, cr661@cam.ac.uk
