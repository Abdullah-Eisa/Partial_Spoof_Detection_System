# Partial_Spoof_Detection_System

Efficient classification and detection of partially faked audio (partial fake speech) using a deep learning pipeline that leverages Wav2Vec 2.0 as a feature extractor and a lightweight back-end classifier (conformer blocks, attention-based pooling, max-pooling and fully-connected layers). The implementation reproduces and extends the approach described in:

Efficient Classification of Partially Faked Audio Using Deep Learning  
Link: https://ieeexplore.ieee.org/document/11130153

Results reported in the paper: EER = 0% (RFP dataset), EER = 2.99% (ASVspoof 2019 LA).

---

Contents
- Project overview
- Repository layout
- Quick start (setup & run)
- Configuration (how to change behavior)
- Running training and inference
- Outputs, model checkpoints and logs
- Troubleshooting & common errors
- Development notes & recommendations
- Citation

---

Project overview
- Purpose: Detect partially manipulated audio segments (partial fake speech) using deep features from Wav2Vec 2.0 and a back-end binary classifier.
- Major components:
  - Feature extractor: Wav2Vec 2.0 (loaded through s3prl/hub)
  - Back-end classifier: Conformer blocks + attention / pooling + FC layers (BinarySpoofingClassificationModel)
  - Data handling: dataset loaders for PartialSpoof, ASVspoof2019, RFP
  - Training loop, validation (dev) evaluation, inference utilities
  - Optional Weights & Biases (WandB) integration for experiments / sweeps

Repository layout (relevant files)
- main.py — entry point for training (uses ConfigManager)
- inference.py — script / functions for running model inference
- train.py — training orchestration and train loop
- preprocess.py — Dataset classes, dataloaders and transforms
- model.py — network architecture (back-end classifier)
- utils/
  - utils.py — project utility functions (I/O, EER computation, wandb helpers, early stopping)
  - config_manager.py — loads config/default_config.yaml and converts ${BASE_DIR} to absolute paths
- config/default_config.yaml — default configuration (data, model, training, inference, paths)
- models/back_end_models/ — checkpoint output (created at training)
- database/ — (expected structure for datasets / labels)
- scripts mentioned in repo (download_database.sh, download_pretrained_model.sh, env_setup.sh, cloud_instance_setup.sh)

---

Quick start (recommended)
1. Clone the repo
   git clone -b main https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
   cd Partial_Spoof_Detection_System

2. Prepare environment
   - Preferred: use the provided env_setup.sh if present to create a conda environment:
     bash env_setup.sh
   - Or install manually:
     python -m venv .venv
     source .venv/bin/activate
     pip install -r requirements.txt
   Required packages (examples): torch, torchaudio, s3prl, tqdm, scikit-learn, pyyaml, wandb (optional)

3. Download data and pretrained feature extractor
   - Use the repository scripts if available:
     bash download_database.sh
     bash download_pretrained_model.sh
   - If those scripts are not used, create the following expected layout or update config:
     /root/Partial_Spoof_Detection_System/config/default_config.yaml controls paths.
     Typical structure under project root:
       database/PartialSpoof/database/train/con_wav
       database/PartialSpoof/database/dev/con_wav
       database/PartialSpoof/database/eval/con_wav
       database/utterance_labels/PartialSpoof_LA_cm_train_trl.json
       database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json
       database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json
     And put the SSL model checkpoint under:
       models/w2v_large_lv_fsh_swbd_cv.pt

4. Configure project
   - Edit config/default_config.yaml to set dataset paths, device, model path, wandb usage and inference options.
   - ConfigManager in utils/config_manager.py replaces ${BASE_DIR} automatically.

5. (Optional) Provide WandB API key
   - Either set environment variable WANDB_API_KEY or create:
     config/wandb_key.txt (single line containing the API key)
   - Enable WandB in config/default_config.yaml (training.use_wandb: true)

---

Configuration (config/default_config.yaml)
- training:
  - num_epochs, learning_rate, batch_size, save_interval, patience, max_grad_norm, monitor_dev_epoch, use_wandb
- model:
  - feature_dim, num_heads, hidden_dim, max_dropout, depthwise_conv_kernel_size, conformer_layers, max_pooling_factor
- data:
  - dataset_name (PartialSpoof_Dataset / ASVspoof2019 / RFP_Dataset)
  - base_path, train_data_path, train_labels_path, dev_data_path, dev_labels_path, eval_data_path, eval_labels_path
- paths:
  - ssl_checkpoint (Wav2Vec2 / SSL model)
  - ps_model_checkpoint (pretrained back-end model for inference)
  - model_save_dir (where training checkpoints are stored)
- system:
  - num_workers, pin_memory, save_feature_extractor, device
- inference:
  - use_cuda, batch_size, apply_transform, num_workers, pin_memory
- wandb_sweep:
  - sweep parameter search space (optional)

Notes:
- ConfigManager replaces ${BASE_DIR} with the repo root. Use absolute or ${BASE_DIR}-relative paths in config.
- If a file/directory in config is missing the code prints warnings; update config paths to match your environment.

---

Running training
- Standard training (uses config/default_config.yaml):
  python main.py
- If wandb enabled (training.use_wandb: true), main.py will try to login using config.get_wandb_key() or your environment key and may start a sweep.
- Trained model checkpoint will be saved under the configured model_save_dir (default: models/back_end_models).
- Watch for FileNotFoundError for label files — ensure dataset and label files exist and paths in config are correct.

Running inference
- Edit config/default_config.yaml inference section and paths.paths.ps_model_checkpoint to point to the saved back-end model.
- Run:
  python inference.py
- The script loads the SSL extractor, instantiates back-end model (BinarySpoofingClassificationModel), loads the specified checkpoint, and runs evaluation on eval_data_path.

Example common errors and fixes
- FileNotFoundError: utterance_labels/...json
  - Ensure label JSON paths from config exist. If you used download scripts, verify their output directory matches config paths or update the config.
- NameError / undefined path variables
  - Confirm config keys match usage in train.py / inference.py (e.g., train_data_path vs train_data).
  - ConfigManager expects keys like train_data_path / train_labels_path / eval_data_path / eval_labels_path — align code or config accordingly.
- WandB warnings:
  - If your session started earlier, rerun wandb login --relogin or set use_wandb to false.

Outputs
- Model checkpoints: models/back_end_models/*.pth
- WandB runs: local wandb/ and remote project if configured
- (Optional) saved metrics / prediction files: outputs/ or saved by util functions

Development notes & recommended improvements
- Move long utility functions into multiple small files (utils/io.py, utils/metrics.py, utils/training.py) for maintainability.
- Replace prints with Python logging (configurable log level).
- Add unit tests for ConfigManager, EER computation, and Dataset loaders.
- Add a requirements.txt and Dockerfile for reproducible environments.
- Validate config on startup and fail-fast with clear error messages.
- Consider adding CLI flags (argparse / click) to override config/default_config.yaml without editing files.

License & contributing
- Please add a LICENSE file to the repo (MIT recommended for permissive reuse).
- Contributions: open issues/PRs; include unit tests for new functionality.

Citation
If you use this code or the approach in academic work, please cite:

Efficient Classification of Partially Faked Audio Using Deep Learning  
BibTeX (suggested):
@inproceedings{eisa2025efficient,  
  title={Efficient Classification of Partially Faked Audio Using Deep Learning},  
  author={Eisa, Abdullah and ...},  
  booktitle={Proceedings ...},  
  year={2025},  
  url={https://ieeexplore.ieee.org/document/11130153}  
}

Contact
- Repository author: Abdullah (see repository for contact details)
- For issues: open an issue on the GitHub repository.

---

Last updated: 2025-08