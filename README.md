# Partial Spoof Detection System

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.8+-orange.svg)](https://pytorch.org/get-started/previous-versions/)

A deep learning-based system for the **efficient classification of partially faked audio** (partial spoof speech). The project leverages **Wav2Vec 2.0** as a feature extractor and a lightweight back-end classifier built with **conformer blocks, attention-based pooling, max pooling, and fully connected layers**.  

This implementation reproduces and extends the methodology from the paper:  
**[Efficient Classification of Partially Faked Audio Using Deep Learning](https://ieeexplore.ieee.org/document/11130153)**
 
---

## 📖 Table of Contents
1. [Overview](#overview)  
2. [Repository Structure](#repository-structure)  
3. [Installation](#installation)  
4. [Download data & pre-trained model (commands)](#download-data--pre-trained-model-commands)  
5. [Configuration](#configuration)  
6. [Training](#training)  
7. [Inference](#inference)  
8. [Outputs](#outputs)  
9. [Troubleshooting](#troubleshooting)  
<!-- 10. [Development Notes](#development-notes)   -->
10. [License](#license)  
11. [Citation](#citation)  
12. [Contact](#contact)  

---

## 1. Overview
- **Problem:** Detect partially manipulated speech where only segments of an audio file are altered/synthesized.  
- **Approach:**  
  - Extract features using **Wav2Vec 2.0** (via `s3prl/hub`).  
  - Classify with **BinarySpoofingClassificationModel**, which combines conformer layers, attention-based pooling, and fully connected classifiers.  
- **Datasets Supported:**  
  - [PartialSpoof](https://github.com/nii-yamagishilab/PartialSpoof)  
  - [ASVspoof2019](https://www.asvspoof.org/)  
  - [RFP Dataset](https://zenodo.org/records/14675126)  
- **Key Results (from paper):**  
  - **0% EER** on RFP dataset  
  - **2.99% EER** on ASVspoof 2019 LA  

---

## 2. Repository Structure

```bash
Partial\_Spoof\_Detection\_System/
├── config/
│   ├── default\_config.yaml      # Main project configuration
│   └── wandb\_key.txt            # (Optional) WandB API key
├── database/                    # Expected dataset storage
│   ├── PartialSpoof/
│   ├── ASVspoof2019/
│   └── RFP/
├── models/
│   └── back\_end\_models/         # Saved checkpoints
├── utils/
│   ├── config\_manager.py        # Loads YAML config & resolves paths
│   ├── utils.py                 # I/O, metrics (EER), WandB helpers
│   └── ...
├── main.py                      # Training entry point
├── inference.py                 # Inference script
├── train.py                     # Training orchestration
├── preprocess.py                # Dataset loaders & transforms
├── model.py                     # Model architecture
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

---

## 3. Installation

### Clone Repository
```bash
git clone https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
cd Partial_Spoof_Detection_System
```

### Environment Setup

Option 1 (script):

```bash
bash env_setup.sh
```

Option 2 (manual):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies:** `torch`, `torchaudio`, `s3prl`, `tqdm`, `scikit-learn`, `pyyaml`, `wandb` (optional)

---

## 4. Download data & pre-trained model (commands)

This repository includes helper scripts to download datasets and the pre-trained Wav2Vec 2.0 checkpoint. Use them after cloning and setting up the environment.

- Download datasets interactively (PartialSpoof, RFP, or ASVspoof2019):
```bash
bash download_database.sh
```
Description: Runs an interactive prompt to choose which dataset to download:
- Option 1 — RFP_database: downloads and unpacks the RFP database zip from Zenodo and arranges files under `database/RFP/`.
- Option 2 — PartialSpoof: downloads PartialSpoof archives (train, segment_labels_v1.2, dev, protocols, eval) from Zenodo and extracts them under `database/PartialSpoof/`. The script also cleans label resolutions other than 0.64.
- Option 3 — ASVspoof2019 LA: downloads the LA partition and extracts it under `database/ASVspoof2019/`.

- Non-interactive example (download PartialSpoof without prompt):
```bash
# choose option 2 automatically
echo "2" | bash download_database.sh
```
Description: Pipes the choice number to the script to automate selection in non-interactive shells.

- Download pre-trained Wav2Vec 2.0 checkpoint:
```bash
bash download_pretrained_model.sh
```
Description: Downloads `w2v_large_lv_fsh_swbd_cv.pt` into the `models/` directory (used as the SSL feature extractor). The script checks for existence before downloading.

Notes:
- Verify free disk space before downloading large datasets.
- After download, ensure the directory layout matches the paths in `config/default_config.yaml` or update the config accordingly.

---

## 5. Configuration

The system is configured via `config/default_config.yaml`. Key sections:

* **training:** epochs, lr, batch size, wandb usage
* **model:** conformer depth, dropout, feature dimensions
* **data:** dataset paths & labels
* **paths:** pretrained checkpoints (Wav2Vec2 + backend model)
* **system:** device, num_workers, memory pinning
* **inference:** batch size, CUDA usage

`ConfigManager` automatically replaces `${BASE_DIR}` with the project root.

---

## 6. Training

Run:

```bash
python main.py
```

or

```bash
python main.py 2>&1 | tee outputs/output.log
```

* Logs and checkpoints → `models/back_end_models/`
* Optional WandB logging if enabled in config.

---

## 7. Inference

Edit `config/default_config.yaml` → set `paths.ps_model_checkpoint` to trained model.
Run:

```bash
python inference.py
```

---

## 8. Outputs

* **Checkpoints:** `.pth` files under `models/back_end_models/`
* **WandB runs:** if enabled, local `wandb/` + online dashboard
* **Metrics:** EER, accuracy, confusion matrix (via utils)

---

## 9. Troubleshooting

* **`FileNotFoundError` for labels** → Ensure dataset/labels exist at config paths.
* **Config mismatches** → Check train/inference scripts expect keys like `train_data_path`.
* **WandB errors** → Re-login (`wandb login --relogin`) or disable by setting `training.use_wandb: false`.

<!-- ---

## 10. Development Notes

* Modularize utils (io, metrics, training).
* Replace prints with structured logging.
* Add unit tests for config, dataset loaders, and metrics.
* Add Dockerfile for reproducibility.
* CLI flags to override YAML configs (via `argparse` or `click`). -->

---

## 10. License

This project is licensed under the **MIT License** – see [LICENSE](LICENSE).

---

## 11. Citation

If you use this work, please cite:

```bibtex
@inproceedings{eisa2025efficient,
  title={Efficient Classification of Partially Faked Audio Using Deep Learning},
  author={Abdulazeez AlAli; George Theodorakopoulos; Abdullah Emad},
  booktitle={Proceedings ...},
  year={2025},
  url={https://ieeexplore.ieee.org/document/11130153}
}
```

---

## 12. Contact

* **Author:** Abdulazeez AlAli; George Theodorakopoulos; Abdullah Emad  
* **Issues:** Please open a GitHub issue for bug reports or feature requests.
