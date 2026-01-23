from __future__ import absolute_import
from __future__ import print_function
import os
import numpy as np
import torch
import wandb
from sklearn.metrics import roc_curve
import json
from sklearn.metrics import roc_auc_score

from datetime import datetime

# Define some utility functions used in the project
# def create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss):
#     metrics_dict=dict()
#     metrics_dict['utterance_eer']=utterance_eer
#     metrics_dict['utterance_eer_threshold']=utterance_eer_threshold
#     metrics_dict['epoch_loss']=epoch_loss
#     return metrics_dict
def create_metrics_dict(utterance_eer, utterance_eer_threshold, epoch_loss, 
                        precision=None, recall=None, f1=None, auc=None):
    """
    Create a comprehensive metrics dictionary
    
    Args:
        utterance_eer: Equal Error Rate
        utterance_eer_threshold: EER threshold
        epoch_loss: Loss for the epoch
        precision: Precision score (optional)
        recall: Recall score (optional)
        f1: F1 score (optional)
        auc: AUC score (optional)
    
    Returns:
        Dictionary containing all metrics
    """
    metrics_dict = dict()
    metrics_dict['utterance_eer'] = utterance_eer
    metrics_dict['utterance_eer_threshold'] = utterance_eer_threshold
    metrics_dict['epoch_loss'] = epoch_loss
    if precision is not None:
        metrics_dict['precision'] = precision
    if recall is not None:
        metrics_dict['recall'] = recall
    if f1 is not None:
        metrics_dict['f1'] = f1
    if auc is not None:
        metrics_dict['auc'] = auc
    return metrics_dict



def load_json_dictionary(path):
  # Load the dictionary from the JSON file
  with open(path, 'r') as json_file:
      my_dict = json.load(json_file)

  return my_dict

def load_labels_txt2dict(path):
    # labels_dict = {}
    label_map = {"spoof": 1, "bonafide": 0}
    labels_dict = dict()
    # file_list=[]
    with open(path, 'r') as f:
        file_lines = f.readlines()
    for line in file_lines:
        line = line.strip()
        if not line:continue  # Skip empty lines

        try:
            _, key, _, _, label = line.split(' ')
            labels_dict[key] = label_map[label]
        except ValueError:
            # If there are not exactly 5 values, print a warning
            print(f"Warning: Skipping malformed line: {line}")
    
    return labels_dict


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        # Convert numpy array to a list
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        # Convert numpy float to a native Python float
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        # Convert numpy int to a native Python int
        return int(obj)
    elif isinstance(obj, dict):
        # Recursively convert dictionary values
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively convert list items
        return [convert_to_serializable(i) for i in obj]
    else:
        # Return the object if it is already serializable
        return obj


def save_json_dictionary(path,my_dict):
  import json

  try:
      with open(path, 'w') as json_file:
          # Convert dictionary to serializable format
          serializable_dict = convert_to_serializable(my_dict)
          json.dump(serializable_dict, json_file, indent=4)
      print(f"Dictionary saved to {path}")
  except PermissionError:
      print(f"Error: Permission denied to write to the file {path}.")
  except IOError as e:
      print(f"Error: {e}")


def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path='checkpoint.pth'):
    # Check if the file exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found at {path}")

    # Load the checkpoint
    checkpoint = torch.load(path)

    # Verify the checkpoint contains the necessary keys
    if 'model_state_dict' not in checkpoint or 'optimizer_state_dict' not in checkpoint or 'epoch' not in checkpoint:
        raise KeyError(f"Checkpoint file is missing required keys ('model_state_dict', 'optimizer_state_dict', 'epoch')")

    # Check if the model state_dict is not empty
    if not checkpoint['model_state_dict']:
        raise ValueError(f"Model state_dict is empty in the checkpoint file at {path}")

    # Load the model and optimizer state dicts
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return the model, optimizer, and epoch number
    return model, optimizer, checkpoint['epoch']

# Code adapted from: https://github.com/YuanGongND/python-compute-eer
def compute_eer(predictions, labels):
    predictions = predictions.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    if labels.ndim > 1 and labels.shape[0] == predictions.shape[0]:
        raise ValueError("labels dimension > 1, 1D vector is only supported for EER computation")
    else:
        # Compute false positive rate (fpr), true positive rate (tpr), and thresholds
        fpr, tpr, thresholds = roc_curve(labels, predictions)
        
        # False Rejection Rate (FRR) is equal to 1 - TPR
        fnr = 1 - tpr

        # Check for NaN values
        if np.any(np.isnan(fnr)) or np.any(np.isnan(fpr)):
            raise ValueError("NaN values found in fnr or fpr. Cannot compute EER.")

        # Find the threshold where fpr (FAR) and frr are closest
        eer_threshold_index = np.nanargmin(np.abs(fpr - fnr))
        eer = (fpr[eer_threshold_index] + fnr[eer_threshold_index]) / 2  # EER is the point where FAR ≈ FRR
        
        # EER value and threshold where it occurs
        eer_threshold = thresholds[eer_threshold_index]
        
        return eer, eer_threshold


# ===========================================================================================================================
# ===========================================================================================================================
# Modularized helper functions

def initialize_wandb():
    """Initialize Weights & Biases for logging"""
    # wandb.init(project='partial_spoof_Wav2Vec2_Conformer_binary_classifier')
    wandb.init()


def compute_metrics(outputs, labels):
    """Compute and return EER and other metrics"""
    utterance_eer, utterance_eer_threshold = compute_eer(outputs, labels)
    return utterance_eer, utterance_eer_threshold


# def log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, backend_model_lr, feature_extractor_lr, dropout_prob, dev_metrics_dict=None):
#     """Log metrics to W&B"""
#     if dev_metrics_dict:
#         wandb.log({
#             'epoch': epoch + 1,
#             'training_loss_epoch': epoch_loss,
#             'training_utterance_eer_epoch': utterance_eer,
#             'training_utterance_eer_threshold_epoch': utterance_eer_threshold,
#             'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
#             'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
#             'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold'],
#             'feature_extractor_lr': feature_extractor_lr,
#             'backend_model_lr': backend_model_lr,
#             'dropout_prob': dropout_prob,
#         })
#     else:
#         wandb.log({
#             'epoch': epoch + 1,
#             'training_loss_epoch': epoch_loss,
#             'training_utterance_eer_epoch': utterance_eer,
#             'training_utterance_eer_threshold_epoch': utterance_eer_threshold,
#             'feature_extractor_lr': feature_extractor_lr,
#             'backend_model_lr': backend_model_lr,
#             'dropout_prob': dropout_prob,
#         })


def compute_precision_recall_f1(predictions, labels, threshold=0.5):
    """
    Safely compute Precision, Recall, and F1 score with edge case handling
    
    Args:
        predictions: Tensor of model predictions (logits or probabilities)
        labels: Tensor of ground truth labels (binary: 0 or 1)
        threshold: Classification threshold (default 0.5)
    
    Returns:
        Tuple of (precision, recall, f1) - all float values
        Returns (0.0, 0.0, 0.0) if unable to compute (edge cases)
    """
    # Convert to numpy and flatten if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Flatten arrays
    predictions = predictions.flatten()
    labels = labels.flatten()
    

    # Convert logits → probabilities if needed
    if np.all((predictions >= 0) & (predictions <= 1)):
        probs = predictions
    else:
        probs = 1 / (1 + np.exp(-predictions))  # sigmoid


    # Convert predictions to binary (0 or 1) using threshold
    # If predictions are probabilities, apply sigmoid; if logits, apply sigmoid
    if np.all((predictions >= 0) & (predictions <= 1)):
        # Predictions are already probabilities
        predicted_labels = (predictions >= threshold).astype(int)
    else:
        # Predictions are logits, apply sigmoid
        predicted_labels = (1 / (1 + np.exp(-predictions)) >= threshold).astype(int)
    
    # Calculate True Positives, False Positives, False Negatives
    tp = np.sum((predicted_labels == 1) & (labels == 1))
    fp = np.sum((predicted_labels == 1) & (labels == 0))
    fn = np.sum((predicted_labels == 0) & (labels == 1))
    
    # Calculate Precision with edge case handling
    if (tp + fp) == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    # Calculate Recall with edge case handling
    if (tp + fn) == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    # Calculate F1 with edge case handling
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    

    # AUC (requires both classes present)
    try:
        if len(np.unique(labels)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0


    return float(precision), float(recall), float(f1), float(auc)



def log_metrics_to_wandb(epoch, epoch_loss, utterance_eer, utterance_eer_threshold, backend_model_lr, feature_extractor_lr, dropout_prob, dev_metrics_dict=None, train_precision=None, train_recall=None, train_f1=None, train_auc=None):
    """
    Log metrics to W&B including Precision, Recall, F1, and AUC scores
    
    Args:
        epoch: Epoch number
        epoch_loss: Training loss for the epoch
        utterance_eer: Training EER
        utterance_eer_threshold: Training EER threshold
        backend_model_lr: Learning rate of backend model
        feature_extractor_lr: Learning rate of feature extractor
        dropout_prob: Dropout probability
        dev_metrics_dict: Dictionary with validation metrics (optional)
        train_precision: Training precision (optional)
        train_recall: Training recall (optional)
        train_f1: Training F1 score (optional)
        train_auc: Training AUC score (optional)
    """
    log_dict = {
        'epoch': epoch + 1,
        'training_loss_epoch': epoch_loss,
        'training_utterance_eer_epoch': utterance_eer,
        'training_utterance_eer_threshold_epoch': utterance_eer_threshold,
        'feature_extractor_lr': feature_extractor_lr,
        'backend_model_lr': backend_model_lr,
        'dropout_prob': dropout_prob,
    }
    
    # Add training precision, recall, f1, auc if provided
    if train_precision is not None:
        log_dict['training_precision_epoch'] = train_precision
    if train_recall is not None:
        log_dict['training_recall_epoch'] = train_recall
    if train_f1 is not None:
        log_dict['training_f1_epoch'] = train_f1
    if train_auc is not None:
        log_dict['training_auc_epoch'] = train_auc
    
    # Add validation metrics if provided
    if dev_metrics_dict:
        log_dict['validation_loss_epoch'] = dev_metrics_dict['epoch_loss']
        log_dict['validation_utterance_eer_epoch'] = dev_metrics_dict['utterance_eer']
        log_dict['validation_utterance_eer_threshold_epoch'] = dev_metrics_dict['utterance_eer_threshold']
        
        # Add validation precision, recall, f1, auc if provided
        if 'precision' in dev_metrics_dict:
            log_dict['validation_precision_epoch'] = dev_metrics_dict['precision']
        if 'recall' in dev_metrics_dict:
            log_dict['validation_recall_epoch'] = dev_metrics_dict['recall']
        if 'f1' in dev_metrics_dict:
            log_dict['validation_f1_epoch'] = dev_metrics_dict['f1']
        if 'auc' in dev_metrics_dict:
            log_dict['validation_auc_epoch'] = dev_metrics_dict['auc']
    
    wandb.log(log_dict)





class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, verbose=False, path=os.path.join(os.getcwd(),'models/back_end_models/RFP_best_model.pth')):
        """
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path to save the best model.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_loss - val_loss > self.delta and val_loss < 0.15:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                # print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
                print(f'Validation utterance_eer decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
            # self.best_model_wts = model.state_dict()  # Save best model weights

            base_path = self.path.split(".")[0]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f"{base_path}_{timestamp}.pth"
            print("==================================================================")
            print(f"in EarlyStopping: model_path= {model_path}")
            # torch.save(self.best_model_wts, model_path)  # Save the model checkpoint
            torch.save({'model_state_dict': model.state_dict()}, model_path)
        else:
            self.counter += 1
            if self.verbose:
                # print(f'Validation loss did not improve. Counter: {self.counter}/{self.patience}')
                print(f'Validation utterance_eer did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")



from typing import List
from huggingface_hub import hf_hub_download

def download_huggingface_hub_datasets(
    files: List[str] = ["database.zip", "LA.zip"],
    repo_id: str = "alsuhba/Rfp_Test",
    repo_type: str = "dataset",
    local_dir: str = "./database/Rfp_Test",
) -> None:
    """
    Download specific files from a Hugging Face dataset repository.
    """
    for filename in files:
        try:
            print(f"Downloading {filename}...")
            hf_hub_download(
                repo_id=repo_id,
                repo_type=repo_type,
                filename=filename,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

# download_huggingface_hub_datasets(files=["database.zip", "LA.zip"])

