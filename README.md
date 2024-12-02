
# Temporal Action Segmentation with Human Skeletons and Attention Mechanism

This project focuses on **temporal action segmentation** using **human skeleton (graph representation)** and attention mechanisms. The model processes **3D human skeleton data** (either real or extracted from RGB videos) to predict an action for each frame, achieving sequence segmentation.

## Table of Contents
- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

---

## Features
- Input: **3D human skeleton data**
  - Real human skeleton data
  - Extracted skeletons from RGB videos
- Output: **Action predictions** for each frame, with actions segmented by classes.
- Supports benchmark datasets:
  - **InHARD**
  - **IKEA Assembly Dataset**
  - **HA4M**
- Evaluation metrics:
  - **Mean Over Frame**
  - **Edit Score**
  - **F1 Score** at thresholds [10%, 25%, 50%].

---

## Datasets
### Links to Datasets:
- **[InHARD Benchmark](https://paperswithcode.com/dataset/inhard)**
- **[IKEA Assembly Dataset](https://ikeaasm.github.io/)**
- **[HA4M Dataset](https://baltig.cnr.it/ISP/ha4m)**

### Data Folder Structure:
- `groundtruth/`: Ground truth annotations for the datasets.
- `position_features/`: Extracted skeleton features.
- `mapping.csv`: Action class mapping.
- `results/`: Model predictions and metrics.

---

## Project Structure
Key files and folders:
- `train.py`: Main script for training the model.
- `eval.py`: Script for model evaluation.
- `model/`: Directory containing different model implementations (e.g., GCN, AGCN, ST-GCN, etc.).
- `helpers/`: Utility functions and helper scripts.
- `requirements.txt`: List of dependencies for the project.

---

## Setup
### Prerequisites
- Python 3.8+
- CUDA-compatible GPU

### Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training
To train the model:
```bash
python train.py --dataset_name <InHARD/IKEA/HA4M> --cudad <cuda_device_number> --base_dir <data_directory_for_dataset> --split <split_number>
```

#### Arguments:
- `--dataset_name`: Name of the dataset to use (e.g., `InHARD`, `IKEA`, `HA4M`).
- `--cudad`: CUDA device number.
- `--base_dir`: Path to the dataset directory.
- `--split`: Dataset split number.

### Evaluation
To evaluate the model:
```bash
python eval.py --dataset_name <gtea/IKEA/HA4M> --cudad <cuda_device_number> --base_dir <data_directory_for_dataset>
```

---

## Results
Predictions and evaluation metrics are stored in the `results/` folder. Metrics include:
- **Mean Over Frame**
- **Edit Score**
- **F1 Score** at thresholds (10%, 25%, 50%).

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
