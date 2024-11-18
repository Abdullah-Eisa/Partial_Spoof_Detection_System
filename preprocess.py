
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # For progress bar (optional)

class RawAudioDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (str): Path to the directory containing the audio files.
            save_dir (str): Path to the directory where the extracted features will be saved.
            tokenizer (callable): A tokenizer for preprocessing the audio data.
            feature_extractor (callable): Feature extractor model (e.g., from HuggingFace).
            transform (callable, optional): Optional transform to apply to the waveform.
            normalize (bool, optional): Whether to normalize the extracted features. Default is True.
        """
        self.directory = directory
        # self.save_dir = save_dir
        # self.tokenizer = tokenizer
        # self.feature_extractor = feature_extractor
        self.transform = transform
        # self.normalize = normalize
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

        # Ensure the save directory exists
        # os.makedirs(save_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.directory, file_name)

        try:
            waveform, sample_rate = torchaudio.load(file_path)
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None  # Or handle the error differently

        if self.transform:
            waveform = self.transform(waveform)


        # # Return a dictionary with the saved feature path and the original file name
        # return {'features': features, 'file_name': file_name, 'save_path': save_path}
        return {'waveform': waveform,'sample_rate': sample_rate, 'file_name': file_name}


def extract_features_with_dataloader(directory, save_dir, tokenizer, feature_extractor, normalize=True,batch_size=32, shuffle=True, num_workers=2, prefetch_factor=2,DEVICE='cpu'):
    """
    Extracts features from audio files in a directory using a DataLoader and saves the features to the specified directory.
    
    This function loads audio data using a custom dataset, processes it in batches, extracts features using 
    the provided feature extractor, and optionally normalizes and saves the resulting features to disk. 
    The processing leverages DataLoader to handle batching, shuffling, and parallel data loading.

    Args:
        directory (str): Path to the directory containing the audio files.
        save_dir (str): Path to the directory where the extracted features will be saved.
        tokenizer (callable): A tokenizer for preprocessing the audio data (e.g., converts waveforms to model-compatible input).
        feature_extractor (callable): Feature extractor model (e.g., from HuggingFace) to process tokenized input and extract features.
        batch_size (int, optional): Number of samples per batch. Default is 32.
        shuffle (bool, optional): Whether to shuffle the data at the beginning of each epoch. Default is True.
        num_workers (int, optional): Number of subprocesses to use for data loading. Default is 2.
        prefetch_factor (int, optional): Number of batches to prefetch per worker. Default is 2.
        normalize (bool, optional): Whether to normalize the extracted features (i.e., L2 normalization along the feature dimension). Default is True.
        DEVICE (str, optional): The device to run the feature extraction on ('cpu' or 'cuda'). Default is 'cpu'.
        
    Returns:
        List of dictionaries: A list of dictionaries, each containing the saved paths and filenames for each batch of features.
        
    Notes:
        - The feature extractor is assumed to return a `last_hidden_state` containing the extracted features.
        - Features are saved with the same filenames as the audio files, with a `__features.pt` suffix added.
        - If `num_workers > 0`, the multiprocessing start method is set to 'spawn' to avoid pickling issues.
        - The `pin_memory` argument is enabled in the DataLoader for faster transfer to GPU.
    """

     
    # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
    if num_workers > 0:
        mp.set_start_method('spawn', force=True)
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Create the dataset instance
    dataset = RawAudioDataset(directory)
    
    # Create the DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=True,  # Enable page-locked memory for faster data transfer to GPU
        prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
        # collate_fn=collate_fn  # Custom collate function to handle variable-length inputs
    )


    # loader_iter = iter(data_loader) # preloading starts here

    # with the default prefetch_factor of 2, 2*num_workers=16 batches will be preloaded
    # the max index printed by __getitem__ is thus 31 (16*batch_size=32 samples loaded)

    # data = next(loader_iter) # this will consume a batch and preload the next one from a single worker to fill the queue
    # batch_size=2 new samples should be loaded

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="data Batches", leave=False):
            waveforms = batch['waveform'].to(DEVICE)
            sample_rates = batch['sample_rate'].to(DEVICE)
            file_names = batch['file_name']
        # for i in range(len(data_loader)):
        #     data = next(loader_iter)
        #     waveforms = data['waveform'].to(DEVICE)
        #     sample_rates = data['sample_rate'].to(DEVICE)
        #     file_names = data['file_name']

            # Tokenize the waveform
            inputs = tokenizer(waveforms.squeeze().numpy(), sampling_rate=sample_rates, return_tensors="pt", padding="longest")

            # Extract features using the feature extractor
            # with torch.no_grad():
            waveforms_features = feature_extractor(**inputs).last_hidden_state.squeeze(0)

            # Normalize features if required
            if normalize:
                waveforms_features = F.normalize(waveforms_features, dim=1)

            # Save the features locally with the same file name and '_features' suffix
            for file_name,waveform_features in zip(file_names,waveforms_features):
                save_path = os.path.join(save_dir, file_name.split('.')[0] + "__features.pt")
                torch.save(waveform_features, save_path)









# import torch.multiprocessing as mp
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

# if __name__ == '__main__':
#     mp.set_start_method('spawn', force=True)  # Ensure spawn method is used


#     use_cuda= True
#     use_cuda =  use_cuda and torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if use_cuda else "cpu")
#     print(f'device: {DEVICE}')

#     BASE_DIR = os.getcwd()

#     train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train3')
#     train_features_path=os.path.join(BASE_DIR,'database\\features\\train')

#     # train_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\train_seglab_0.64.npy')
#     # train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


#     Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
#     Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
#     Wav2Vec2_model.eval()


#     # Record the start time
#     start_time = datetime.now()

#     # train_loader = get_audio_data_loaders(train_files_path, train_seglab_64_dict,batch_size=8, shuffle=True)
#     # dataset=RawAudioDataset(train_files_path,train_features_path,Wav2Vec2_tokenizer,Wav2Vec2_model)
#     extract_features_with_dataloader(train_files_path, train_features_path, Wav2Vec2_tokenizer, Wav2Vec2_model, normalize=True,batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2,DEVICE=DEVICE)
 

#     # Record the end time
#     end_time = datetime.now()
#     total_training_time = end_time - start_time
#     print(f"Total training time: {total_training_time}")

#     # Extract hours, minutes, and seconds
#     total_seconds = total_training_time.total_seconds()
#     hours = int(total_seconds // 3600)
#     minutes = int((total_seconds % 3600) // 60)
#     seconds = int(total_seconds % 60)

#     # Print training time in hours, minutes, and seconds
#     print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")




import torch.multiprocessing as mp
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)  # Ensure spawn method is used


    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    BASE_DIR = os.getcwd()

    train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train3')
    train_features_path=os.path.join(BASE_DIR,'database\\features\\train')

    train_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


    # Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    # Wav2Vec2_model.eval()


    # Record the start time
    start_time = datetime.now()

    train_loader = get_audio_data_loaders(train_features_path, train_seglab_64_dict, batch_size=32, shuffle=True, num_workers=2, prefetch_factor=2)
    # dataset=RawAudioDataset(train_files_path,train_features_path,Wav2Vec2_tokenizer,Wav2Vec2_model)
    # extract_features_with_dataloader(train_files_path, train_features_path, Wav2Vec2_tokenizer, Wav2Vec2_model, normalize=True,batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2,DEVICE=DEVICE)
 

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




# ================================================= load saved features ==========================


# class FeaturesAudioDataset(Dataset):
#     def __init__(self, features_dir, labels_dict, normalize=True):
#         """
#         Args:
#             features_dir (str): Directory where pre-saved feature files are stored.
#             labels_dict (dict): Dictionary mapping file names (without extensions) to labels.
#             normalize (bool, optional): Whether to normalize the features. Default is True.
#         """
#         self.features_dir = features_dir
#         self.labels_dict = labels_dict
#         self.normalize = normalize
#         self.file_list = [f for f in os.listdir(features_dir) if f.endswith('__features.pt')]  # Assuming feature files have _features.pt suffix

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         file_name = self.file_list[idx]
#         feature_path = os.path.join(self.features_dir, file_name)

#         # Load the saved features
#         features = torch.load(feature_path)
#         print(f'{file_name} has extracted features of type {type(features)}  with size {features.size()}')

#         # Normalize features if required
#         # if self.normalize:
#         #     features = torch.nn.functional.normalize(features, dim=0)  # Normalize across feature dimension

#         # Extract the base file name (without extension or suffix)
#         base_file_name = file_name.split('__features.pt')[0]  # Assuming feature files are named with '_features.pt' suffix

#         # Fetch the corresponding label using the base file name
#         # label = self.labels_dict.get(base_file_name, -1)  # Default to -1 if label is not found
#         label = self.labels_dict.get(base_file_name).astype(int)
#         # print(f"label={label}")
#         # label = self.labels_dict.get(base_file_name)
#         label = torch.tensor(label, dtype=torch.int8)
#         print(f'{file_name} has label of type {type(label)}  with size {label.size()}, Array={label}')

#         return {'features': features, 'label': label, 'file_name': base_file_name}




# def collate_fn(batch):
#     batch = [item for item in batch if item is not None]  # Remove None values
#     if len(batch) == 0:
#         return None
    
#     features = [item['features'] for item in batch]
#     labels = [item['label'] for item in batch]
    
#     # Pad features to have the same length
#     features_padded = pad_sequence(features, batch_first=True)

#     # Determine the maximum length of labels in the batch
#     max_label_length = 33

#     # Pad labels to the fixed length of 33
#     labels_padded = []
#     for label in labels:
#         # If the label is shorter than the fixed length, pad it
#         if label.size(0) < max_label_length:
#             padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=-1)
#             # padded_label = F.pad(label, (0, max_label_length - label.size(0)), value=float('nan'))
#         else:
#             padded_label = label[:max_label_length]
#         labels_padded.append(padded_label)
    
#     # Stack padded labels to a single tensor
#     labels_padded = torch.stack(labels_padded)

#     return {
#         'features': features_padded,
#         'label': labels_padded,
#         'file_name': [item['file_name'] for item in batch]
#     }



# from torch.utils.data import DataLoader
# import torch.multiprocessing as mp

# # Assuming AudioDataset and collate_fn are defined elsewhere
# # def get_audio_features_loaders(directory, labels_dict, batch_size=32, shuffle=True, num_workers=2, prefetch_factor=2):
# def get_audio_features_loaders(directory, labels_dict, batch_size=32, shuffle=True, num_workers=0, prefetch_factor=None):
    
#     # If multiprocessing is used, set start method to 'spawn' (for avoiding pickling issues)
#     # if num_workers > 0:
#     #     mp.set_start_method('spawn', force=True)
    
#     # Create the dataset instance
#     dataset = FeaturesAudioDataset(directory, labels_dict)
    
#     # Create the DataLoader
#     data_loader = DataLoader(
#         dataset,
#         batch_size=batch_size, 
#         shuffle=shuffle, 
#         num_workers=num_workers, 
#         pin_memory=True,  # Enable page-locked memory for faster data transfer to GPU
#         prefetch_factor=prefetch_factor,  # How many batches to prefetch per worker
#         collate_fn=collate_fn  # Custom collate function to handle variable-length inputs
#     )
    
#     return data_loader









# import os
# from tqdm import tqdm  # Correctly import tqdm
# from datetime import datetime

# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # from transformers import Wav2Vec2Processor, 
# from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

# # import wandb

# from utils import *
# from model import *
# from inference import dev_model





# def features_train_model(train_directory, train_labels_dict, 
#                 BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.001,
#                 model_save_path=os.path.join(os.getcwd(),'models\\back_end_models'),
#                 DEVICE='cpu',save_interval=float('inf')):

#     # Ensure the model save path exists
#     os.makedirs(model_save_path, exist_ok=True)
#     # Load utterance labels
#     BASE_DIR = os.getcwd()
#     PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database\\utterance_labels\\PartialSpoof_LA_cm_train_trl.json')
#     PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

#     # Initialize the model, loss function, and optimizer
#     hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
#     PS_Model = MyModel(d_model=hidd_dims['wav2vec2'],gmlp_layers=5).to(DEVICE)  # Move model to the configured device
    
#     # Wrap the model with DataParallel
#     if torch.cuda.device_count() > 1:
#         PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
#         print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")


#     # criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with Logits for multi-label classification
#     # criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
#     criterion = CustomLoss().to(DEVICE)
#     optimizer = optim.Adam(PS_Model.parameters(), lr=LEARNING_RATE)
    
#     # Get the data loader
#     train_loader = get_audio_features_loaders(train_directory, train_labels_dict, batch_size=BATCH_SIZE, shuffle=True)

#     PS_Model.train()  # Set the model to training mode
    

#     files_names=[]
#     training_segment_eer_per_epoch=[]
#     dev_segment_eer_per_epoch=[]

#     for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
#         PS_Model.train()  # Set the model to training mode

#         epoch_loss = 0
#         epoch_segment_eer = 0
#         epoch_segment_eer_threshold = 0
#         utterance_eer, utterance_eer_threshold=0,0
#         utterance_predictions=[]
#         # utterance_labels=[]

#         for batch in tqdm(train_loader, desc="Train Batches", leave=False):
#             features = batch['features'].to(DEVICE)
#             labels = batch['label'].to(DEVICE)

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Pass features to model and get predictions
#             outputs = PS_Model(features)
#             # print(f"labels type: {type(labels)}, size: {labels.size()} , labels=\n {labels}")
#             # print(f"outputs type: {type(outputs)}, size: {outputs.size()} , outputs=\n {outputs}")

#             # Calculate loss
#             loss = criterion(outputs, labels) 
#             epoch_loss += loss.item()


#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()


#             with torch.no_grad():  # No need to compute gradients for EER calculation

#                 # Calculate utterance predictions

#                 utterance_predictions.extend(get_uttEER_by_seg(outputs,labels))

#                 # Calculate segment EER
#                 batch_segment_eer, batch_segment_eer_threshold = compute_eer(outputs,labels)
#                 epoch_segment_eer += batch_segment_eer
#                 epoch_segment_eer_threshold += batch_segment_eer_threshold


#                 # Accumulate files names
#                 if epoch == 0:
#                     batch_file_names = batch['file_name']
#                     files_names.extend(batch_file_names)


#             # Print batch training progress
#             # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch Loss: {loss.item()}, Batch Segment EER: {segment_eer:.4f}, Batch Segment EER Threshold: {segment_eer_threshold:.4f}')

#         # Save checkpoint
#         if NUM_EPOCHS>=save_interval and (epoch + 1) % (NUM_EPOCHS//save_interval) == 0:
#             # Generate a unique filename based on hyperparameters
#             timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#             model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
                        
#             save_checkpoint(PS_Model, optimizer, epoch + 1,os.path.join(model_save_path,model_filename))

#         # Get Average Utterance EER for the epoch
#         if epoch ==0: utterance_labels =[PartialSpoof_LA_cm_train_trl_dict[file_name] for file_name in files_names]
#         print(f"utterance_labels:\n {utterance_labels}")
#         print(f"utterance_predictions:\n {utterance_predictions}")
#         utterance_predictions = torch.cat(utterance_predictions)
#         utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

#         # Average Segment EER and loss for the epoch
#         epoch_loss /= len(train_loader)
#         epoch_segment_eer /= len(train_loader)
#         epoch_segment_eer_threshold /= len(train_loader)

#         # Print epoch training progress
#         print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete. Average Loss /epoch : {epoch_loss:.4f},\n'
#                f'Average Segment EER: {epoch_segment_eer:.4f}, Average Segment EER Threshold: {epoch_segment_eer_threshold:.4f},\n'
#                f'Average Training Utterance EER: {utterance_eer:.4f}, Average Training Utterance EER Threshold: {utterance_eer_threshold:.4f}')


#         training_segment_eer_per_epoch.append(epoch_segment_eer)

#         BASE_DIR = os.getcwd()
#         # Define development files and labels
#         # dev_files_path=os.path.join(BASE_DIR,'database\\dev\\con_wav')
#         # dev_files_path=os.path.join(BASE_DIR,'database\\mini_database\\dev')
#         # dev_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\dev_seglab_0.64.npy')
#         # dev_seglab_64_dict = np.load(dev_seglab_64_path, allow_pickle=True).item()

#         # dev_metrics_dict=dev_model( PS_Model,dev_files_path, dev_seglab_64_dict, tokenizer,feature_extractor, BATCH_SIZE,DEVICE=DEVICE)
#         # dev_segment_eer_per_epoch.append(dev_metrics_dict['segment_eer'])


#     # plot training EER per epoch
#     plot_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
#     plot_train_dev_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch, dev_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
#     # # plot Vali EER per epoch
#     # plot_eer_per_epoch(NUM_EPOCHS, training_eer_per_epoch)


#     # Generate a unique filename based on hyperparameters
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     model_filename = f"model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
    
#     # Save the trained model
#     model_save_path=os.path.join(model_save_path,model_filename)
#     # torch.save(PS_Model.state_dict(), model_save_path)
#     save_checkpoint(PS_Model, optimizer,NUM_EPOCHS,model_save_path)
#     print(f"Model saved to {model_save_path}")


#     # Save metrics
#     training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_segment_eer,epoch_segment_eer_threshold,epoch_loss)
#     training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
#     training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs\\{training_metrics_dict_filename}')
#     save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

#     if DEVICE=='cuda': torch.cuda.empty_cache()
#     # wandb.finish()
#     print("Training complete!")








# if __name__ == "__main__":

#     # Device configuration
#     use_cuda= True
#     use_cuda =  use_cuda and torch.cuda.is_available()
#     DEVICE = torch.device("cuda" if use_cuda else "cpu")
#     print(f'device: {DEVICE}')

#     BASE_DIR = os.getcwd()

#     # Define training files and labels
#     # train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train')
#     # train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train2')
#     # train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train3')
#     # train_files_path=os.path.join(BASE_DIR,'database\\train\\con_wav')
#     train_features_path=os.path.join(BASE_DIR,'database\\features\\train')
#     train_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\train_seglab_0.64.npy')
#     train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


#     # Load the tokenizer and model from the local directory
#     # Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
#     # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
#     # Wav2Vec2_model.eval()


#     # Record the start time
#     start_time = datetime.now()
#     # train model
#     if torch.cuda.device_count() > 1:  
#         gpu_num = torch.cuda.device_count()
#         BATCH_SIZE=int(count_files_in_directory(train_features_path)/ gpu_num) 
#         print(f"BATCH_SIZE={BATCH_SIZE}")
#     else:
#         BATCH_SIZE=16
#     features_train_model(train_features_path, train_seglab_64_dict, BATCH_SIZE=BATCH_SIZE,NUM_EPOCHS=10,DEVICE=DEVICE)

#     # Record the end time
#     end_time = datetime.now()
#     total_training_time = end_time - start_time
#     print(f"Total training time: {total_training_time}")

#     # Extract hours, minutes, and seconds
#     total_seconds = total_training_time.total_seconds()
#     hours = int(total_seconds // 3600)
#     minutes = int((total_seconds % 3600) // 60)
#     seconds = int(total_seconds % 60)

#     # Print training time in hours, minutes, and seconds
#     print(f"Total training time: {hours} hours, {minutes} minutes, {seconds} seconds")
