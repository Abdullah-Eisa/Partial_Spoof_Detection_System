




# from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForSequenceClassification
# # Load the pre-trained model and processor
# # model_name = "facebook/wav2vec2-conformer-rope-large-960h-ft"  # Example model name
# model_name = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"  # Example model name
# processor = Wav2Vec2Processor.from_pretrained(model_name)
# model = Wav2Vec2ConformerForSequenceClassification.from_pretrained(model_name)

# # Save them locally
# processor.save_pretrained("./models/Wav2Vec2Processor")
# model.save_pretrained("./models/Wav2Vec2ConformerForSequenceClassificationModel")








# from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForSequenceClassification
# import torch
# import torchaudio
# import os

# use_cuda = True
# use_cuda = use_cuda and torch.cuda.is_available()
# DEVICE = torch.device("cuda" if use_cuda else "cpu")

# # Load the pre-trained model and processor
# processor = Wav2Vec2Processor.from_pretrained("models/Wav2Vec2Processor")
# model = Wav2Vec2ConformerForSequenceClassification.from_pretrained("models/Wav2Vec2ConformerForSequenceClassificationModel").to(DEVICE)

# # Load and preprocess the .wav file
# file_name = "CON_T_0000000"
# file_path = os.path.join(os.getcwd(), f'database/train/con_wav/{file_name}.wav')
# waveform, sample_rate = torchaudio.load(file_path, normalize=True)

# # Ensure the waveform has the expected shape
# print(f"Original waveform shape: {waveform.shape}")

# # Remove any extra dimensions if necessary (flatten the waveform if it has an extra dimension)
# waveform = waveform.squeeze()  # Remove dimensions of size 1

# # Now the shape of the waveform should be [1, N], where N is the length of the audio
# print(f"Processed waveform shape: {waveform.shape}")

# # Preprocess the audio input for the model
# inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True).to(DEVICE)

# # Check the type and shape of the inputs
# print(f"inputs type= {type(inputs)}")
# print(f"inputs shape= {inputs.input_values.shape}")

# # Make a prediction
# with torch.no_grad():
#     logits = model(**inputs).logits

# # Get the predicted class
# predicted_class_ids = torch.argmax(logits, dim=-1).item()

# # Map the predicted class id to the label
# predicted_label = model.config.id2label[predicted_class_ids]
# print(f"Predicted label: {predicted_label}")
# # Convert the predicted class ID to binary (0 or 1)
# predicted_binary_label = int(predicted_class_ids)  # Assumes model outputs 0 or 1 for binary classification

# # Print the predicted binary label (0 or 1)
# print(f"predicted_class_ids: {predicted_class_ids}  , Predicted binary label: {predicted_binary_label}")


# =======================================================================================================================================================
# =======================================================================================================================================================
# =======================================================================================================================================================
# =======================================================================================================================================================
# =======================================================================================================================================================


import os
from tqdm import tqdm  # Correctly import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# from transformers import Wav2Vec2Processor, 
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

import wandb

from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForSequenceClassification
import torch
import torchaudio
import os

from sklearn.metrics import roc_curve

# from utils import *
# from model import *
# from model import MyUpdatedSpoofingDetectionModel

# from inference import dev_model


# ... (your training and inference functions)

import os
import torch
from torch.utils.data import Dataset, DataLoader
# import soundfile as sf
from transformers import Wav2Vec2Processor
# from sklearn.model_selection import train_test_split

class AudioDataset(Dataset):
    def __init__(self, dataset_directory, labels, processor, transform=None,normalize=True, sampling_rate=16000):
        """
        Args:
            audio_files (list of str): List of paths to the .wav files.
            labels (list of int): Corresponding labels (0 or 1 for spoof/bonafide).
            processor (Wav2Vec2Processor): The processor for converting audio to input tensors.
            sampling_rate (int): Sampling rate for loading audio.
        """
        self.audio_files = [f for f in os.listdir(dataset_directory) if f.endswith('.wav')]
        self.labels = labels
        self.processor = processor
        self.dataset_directory = dataset_directory
        # self.labels_dict = labels_dict
        self.transform = transform
        self.normalize = normalize
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Load audio
        # audio_input, _ = sf.read(self.audio_files[idx])
        file_name = self.audio_files[idx]
        file_path = os.path.join(self.dataset_directory, file_name)

        try:
            waveform, sampling_rate = torchaudio.load(file_path, normalize=True)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None
        


                # Apply any other transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        # Return raw waveform and sample rate
        # Get the label
        file_name = file_name.split('.')[0]
        # label = self.labels.get(file_name).astype(int)
        label = self.labels.get(file_name,-1)

        # label = torch.tensor(label, dtype=torch.int8)
        # label = torch.tensor([label], dtype=torch.int8)  # Ensure label is 1-dimensional
        label = torch.tensor([label], dtype=torch.int64)  # Ensure label is 1-dimensional
        # label = self.labels[idx]

        # Process audio for model input
        inputs = self.processor(waveform, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

        # Convert the inputs to tensors
        input_values = inputs.input_values.squeeze(0)  # Shape: [sequence_length]
        attention_mask = inputs.attention_mask.squeeze(0)  # Shape: [sequence_length]

        return {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label),
            "file_name": file_name
        }





from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]  # Remove None values
    if len(batch) == 0:
        return None
    
    waveforms = [item['input_values'] for item in batch]
    # labels = [item['labels'] for item in batch]

    # Pad waveforms to have the same length
    waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)

    # Determine the maximum length of labels in the dataset
    # max_label_length = 33

    # # Pad labels to the fixed length of 33
    # labels_padded = []
    # for label in labels:
    #     # If the label is shorter than the fixed length, pad it
    #     if label.size(0) < max_label_length:
    #         padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=-1)
    #         # padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=float('nan'))
    #     else:
    #         padded_label = label[:max_label_length]
    #     labels_padded.append(padded_label)
    
    # # Stack padded labels to a single tensor
    # labels_padded = torch.stack(labels_padded)

    # print(f"[item['labels'] for item in batch]:  {[item['labels'] for item in batch]}")
    return {
        'input_values': waveforms_padded,
        # 'attention_mask': torch.cat([item['attention_mask'] for item in batch], dim=0),
        'attention_mask': torch.cat([item['attention_mask'] for item in batch]),
        'labels': torch.cat([item['labels'] for item in batch]),
        'file_name': [item['file_name'] for item in batch]
    }


# torch.cat(utterance_predictions, dim=0)


import torch.multiprocessing as mp

# Assuming AudioDataset and collate_fn are defined elsewhere
def get_raw_labeled_audio_data_loaders(dataset_directory, labels_dict,processor, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    
    # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)
    
    # Create the dataset instance
    dataset = AudioDataset(dataset_directory, labels_dict,processor)
    
    # pitch_shift_transform = PitchShiftTransform(sample_rate=16000, pitch_shift_prob=0.5, pitch_shift_steps=(-2, 2))

    # # Initialize the dataset with the transform
    # dataset = RawLabeledAudioDataset(
    #     directory=directory,
    #     labels_dict=labels_dict,
    #     transform=pitch_shift_transform  # Apply pitch shift as part of the dataset transform
    # )


    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
        collate_fn=custom_collate_fn  # Custom collate function to handle variable-length inputs
    )
    
    return data_loader




def get_masked_labels_and_outputs(model_output,labels_tensor):
    # Create mask to identify valid labels (not -1)
    mask_tensor = (labels_tensor != -1)
    # print(f"mask_tensor:\n {mask_tensor}")
    # Remove -1 values from labels using the mask
    masked_labels = labels_tensor[mask_tensor]

    # Remove equivalent positions in the output tensor using the mask
    masked_output = model_output[mask_tensor]

    return masked_output,masked_labels 

from sklearn.metrics import roc_curve

def compute_eer(predictions, labels):

    # Mask padding value
    # predictions, labels =get_masked_labels_and_outputs(predictions, labels)
    # print(f"after Mask padding value,\n nontarget_scores=\n{nontarget_scores} target_scores=\n{target_scores} ")
    # print(f"after Masking,\n predictions= {predictions} \n labels= {labels}")
    # Ensure scores and labels are PyTorch tensors and detach them
    # predictions = predictions.detach().cpu().numpy()
    # labels = labels.detach().cpu().numpy()

    # predictions = predictions.numpy()
    # labels = labels.numpy()
    
    # if labels.ndim > 1 and labels.shape[0] == predictions.shape[0]:
    # if labels.shape[0]  > 1 and labels.shape[0] == predictions.shape[0]:
    #     raise ValueError("labels dimension > 1, 1D vector is only supported for EER computation")
    # else:

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



def load_json_dictionary(path):
  import json

  # Define the path to your JSON file
  # input_file_path = os.path.join(BASE_DIR,'PartialSpoof_LA_cm_eval_trl.json')

  # Load the dictionary from the JSON file
  with open(path, 'r') as json_file:
      my_dict = json.load(json_file)

  return my_dict


def save_checkpoint(model, optimizer, epoch, path='checkpoint.pth'):
    # os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def train_model(train_directory, train_labels_dict, 
                BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.0001,
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                DEVICE='cpu',save_interval=float('inf'),patience=10,save_feature_extractor=False):

    # Initialize W&B
    wandb.init(project='partial_spoof_Wav2Vec2Conformer')

    # Initialize early stopping
    # early_stopping = EarlyStopping(patience=patience, verbose=True)

    # if DEVICE == 'cuda':
    torch.cuda.empty_cache()
    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)
    # Load utterance labels
    BASE_DIR = os.getcwd()
    # PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    # PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Load the pre-trained model and processor
    processor = Wav2Vec2Processor.from_pretrained("models/Wav2Vec2Processor")
    PS_Model = Wav2Vec2ConformerForSequenceClassification.from_pretrained("models/Wav2Vec2ConformerForSequenceClassificationModel").to(DEVICE)

    # Initialize the model, loss function, and optimizer
    # hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    # PS_Model = MyUpdatedSpoofingDetectionModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128, num_classes=33,conformer_layers=1).to(DEVICE)  # Move model to the configured device

    # Wrap the model with DataParallel
    # if torch.cuda.device_count() > 1:
    #     PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
    #     print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # optimizer = optim.Adam(PS_Model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
        

    # criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    # criterion = CustomLoss().to(DEVICE)


    gamma=0.9
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Get the data loader
    train_dataloader = get_raw_labeled_audio_data_loaders(train_directory, train_labels_dict,processor ,batch_size=BATCH_SIZE, shuffle=True, num_workers=2, prefetch_factor=2)

    # Logging gradients with wandb.watch
    wandb.watch(PS_Model, log_freq=100)

    PS_Model.train()  # Set the model to training mode

    files_names=[]
    training_segment_eer_per_epoch=[]
    dev_segment_eer_per_epoch=[]

    for epoch in range(NUM_EPOCHS):
        PS_Model.train()
        total_loss = 0
        # correct_predictions = 0
        # total_predictions = 0
        utterance_predictions=[]
        utterance_labels=[]
        c=0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
            # if c>4:
            #     break
            # else:
            #         c+=1

            # break
            # Move inputs and labels to device (GPU or CPU)
            input_values = batch["input_values"].to(DEVICE)
            # attention_mask = batch["attention_mask"].to(DEVICE)
            # attention_mask = batch["attention_mask"]
            labels = batch["labels"].to(DEVICE)
            # print(f"labels: {labels}")
            # print(f"attention_mask: {attention_mask}")
            # print(f"input_values: {input_values}")
            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            # outputs = PS_Model(input_values, attention_mask=attention_mask, labels=labels)
            outputs = PS_Model(input_values, labels=labels)
            logits = outputs.logits

            # Calculate loss
            loss = outputs.loss
            total_loss += loss.item()

            # Get predictions and calculate accuracy
            # predicted_class_ids = int(torch.argmax(logits, dim=-1).item())
            # predicted_class_ids = torch.argmax(logits, dim=-1).item()
            predicted_class_ids = torch.argmax(logits, dim=-1)
            # print(f"logits: {logits}")
            print(f"predicted_class_ids: {predicted_class_ids}")
            # correct_predictions += (predicted_class_ids == labels).sum().item()
            # total_predictions += labels.size(0)

            # Backward pass
            loss.backward()
            optimizer.step()

            # with torch.no_grad():  # No need to compute gradients for EER calculation

            #     utterance_predictions.extend(predicted_class_ids)
            #     utterance_labels.extend(labels)

            with torch.no_grad():  # No need to compute gradients for EER calculation
                utterance_predictions.extend(predicted_class_ids.detach().cpu().numpy())
                utterance_labels.extend(labels.detach().cpu().numpy())


        # Calculate average loss and accuracy for this epoch
        epoch_loss = total_loss / len(train_dataloader)

        # utterance_predictions=torch.cat(utterance_predictions, dim=0)
        # utterance_labels=torch.cat(utterance_labels, dim=0)
        utterance_eer, utterance_eer_threshold= compute_eer(utterance_predictions,utterance_labels)

        # accuracy = correct_predictions / total_predictions
        # print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Validation phase (optional)
        # model.eval()
        # with torch.no_grad():
        #     val_loss = 0
        #     val_correct_predictions = 0
        #     val_total_predictions = 0

        #     for batch in tqdm(val_dataloader, desc="Validation", leave=False):
        #         input_values = batch["input_values"].to(device)
        #         attention_mask = batch["attention_mask"].to(device)
        #         labels = batch["labels"].to(device)

        #         outputs = model(input_values, attention_mask=attention_mask, labels=labels)
        #         logits = outputs.logits

        #         # Calculate loss
        #         val_loss += outputs.loss.item()

        #         # Get predictions and calculate accuracy
        #         predicted_class_ids = torch.argmax(logits, dim=-1)
        #         val_correct_predictions += (predicted_class_ids == labels).sum().item()
        #         val_total_predictions += labels.size(0)

        #     # Calculate average validation loss and accuracy
        #     avg_val_loss = val_loss / len(val_dataloader)
        #     val_accuracy = val_correct_predictions / val_total_predictions
        #     print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            

        wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
            # 'training_segment_eer_epoch': segment_eer, 
            # 'training_segment_eer_threshold_epoch': segment_eer_threshold,
            'training_utterance_eer_epoch': utterance_eer,
            'training_utterance_eer_threshold_epoch': utterance_eer_threshold, 
            # 'training_utterance_pooling_eer_epoch': utterance_pooling_eer,
            # 'training_utterance_pooling_eer_threshold_epoch': utterance_pooling_eer_threshold, 
            # 'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
            # 'validation_segment_eer_epoch': dev_metrics_dict['segment_eer'], 
            # 'validation_segment_eer_threshold_epoch': dev_metrics_dict['segment_eer_threshold'],
            # 'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
            # 'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold']                      
            })
        
        scheduler.step()


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"

    # Save the trained model
    model_save_path=os.path.join(model_save_path,model_filename)
    save_checkpoint(PS_Model, optimizer,NUM_EPOCHS,model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save segment_predictions, segment_labels, utterance_predictions, utterance_labels
    torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

    if DEVICE=='cuda': torch.cuda.empty_cache()
    wandb.finish()
    print("Training complete!")



def train():
    # Initialize W&B
    wandb.init(project='partial_spoof_Wav2Vec2Conformer')

    # Extract parameters from W&B configuration
    config = wandb.config
    
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    train_files_path=os.path.join(BASE_DIR,'database/train/con_wav')
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train3')
    # train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    # train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()
    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Call train_model with parameters from W&B sweep
    train_model(
        train_directory=train_files_path,
        train_labels_dict=PartialSpoof_LA_cm_train_trl_dict,
        BATCH_SIZE=config.BATCH_SIZE,
        NUM_EPOCHS=config.NUM_EPOCHS,
        LEARNING_RATE=config.LEARNING_RATE,
        DEVICE=DEVICE,
        save_interval=10
    )





import wandb
import os
import random
from datetime import datetime

# from train import train

def main():
    """ main(): the default wrapper for training and inference process
    """

    # wandb_key="Get the key here"
    wandb_api_key="c1fc533d0bafe63c83a9110c6daef36b76f77de1"
    # os.system(f"echo {wandb_key}")
    wandb.login(key=wandb_api_key,relogin=True,force=True)

    # wandb.init(project='partial_spoof_demo')


    sweep_config = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'training_utterance_eer_epoch'
            },
        'parameters': 
        {
            # 'NUM_EPOCHS': {'values': [5, 7]},
            # 'LEARNING_RATE': {'values': [0.001]},
            # 'BATCH_SIZE': {'values': [16,32]},
            'NUM_EPOCHS': {'values': [1]},
            'LEARNING_RATE': {'values': [0.002]},
            'BATCH_SIZE': {'values': [8]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config,project='partial_spoof_Wav2Vec2Conformer')
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=1)




if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    main()

    # Record the end time
    end_time = datetime.now()
    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time}")

    # Extract hours, minutes, and seconds
    total_seconds = total_training_time.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Print training time in hours, minutes, and seconds
    print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")

