
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
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

import wandb

from utils import *
from model import *
from inference import dev_model


import torch.distributed as dist

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp




import os
from tqdm import tqdm  # Correctly import tqdm
# from transformers import Wav2Vec2Processor, 
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
import numpy as np

import torch
import torch.optim as optim




class EarlyStopping:
    def __init__(self, patience=10, delta=0, verbose=False, path=os.path.join(os.getcwd(),'models/back_end_models/best_model.pth')):
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
        if self.best_loss - val_loss > self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model...')
            self.best_model_wts = model.state_dict()  # Save best model weights
            torch.save(self.best_model_wts, self.path)  # Save the model checkpoint
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not improve. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print("Early stopping triggered.")


def setup(rank, world_size):
    # Initialize the distributed environment.
    dist.init_process_group(
        backend='nccl',  # Use NCCL for multi-GPU training
        init_method='env://',  # Environment variable-based initialization
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)  # Each process is assigned a specific GPU

def cleanup():
    dist.destroy_process_group()  # Clean up the distributed environment


def train_model(rank,world_size,train_directory, train_labels_dict, 
                BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.0001,
                model_save_path=os.path.join(os.getcwd(),'models/back_end_models'),
                DEVICE='cpu',save_interval=float('inf'),patience=10):

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Initialize the distributed environment
    setup(rank, world_size)

    # Initialize W&B
    wandb.init(project='partial_spoof_trial_2')

    if DEVICE == 'cuda':
        torch.cuda.empty_cache()
    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)
    # Load utterance labels
    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')
    PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Load feature extractor
    Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(rank)
    Wav2Vec2_model.eval()

    # Initialize the model, loss function, and optimizer
    hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    # PS_Model = MyModel(d_model=hidd_dims['wav2vec2'],gmlp_layers=5).to(DEVICE)  # Move model to the configured device
    # PS_Model = SpoofingDetectionModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128, num_classes=33).to(DEVICE)  # Move model to the configured device
    PS_Model = MyUpdatedSpoofingDetectionModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128, num_classes=33).to(rank)  # Move model to the configured device

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        # Wav2Vec2_model = DDP(Wav2Vec2_model, device_ids=[rank])
        PS_Model = DDP(PS_Model, device_ids=[rank])
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    

    # criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    # criterion = CustomLoss()
    criterion = CustomLoss().to(rank)
    optimizer = optim.Adam(PS_Model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(list(PS_Model.parameters()) + list(wav2vec2_model.parameters()), lr=LEARNING_RATE)
    optimizer = optim.AdamW(PS_Model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)
    gamma=0.9
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # Get the data loader
    # Create a DistributedSampler to ensure each process gets a unique subset of data    
    train_loader = get_raw_labeled_audio_data_loaders(rank,world_size,train_directory, train_labels_dict,batch_size=BATCH_SIZE, shuffle=True)

    PS_Model.train()  # Set the model to training mode
    

    files_names=[]
    training_segment_eer_per_epoch=[]
    dev_segment_eer_per_epoch=[]

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        PS_Model.train()  # Set the model to training mode

        epoch_loss = 0
        utterance_eer, utterance_eer_threshold=0,0
        segment_eer, segment_eer_threshold=0,0
        utterance_predictions=[]
        # utterance_pooling_predictions=[]
        segment_predictions=[]
        segment_labels=[]

        for batch in tqdm(train_loader, desc="Train Batches", leave=False):
        # for i in range(len(data_loader)):
        #     data = next(loader_iter)
        #     waveforms = data['waveform'].to(DEVICE)
        #     labels = data['label'].to(DEVICE)
            # waveforms = batch['waveform'].to(DEVICE)
            waveforms = batch['waveform']
            labels = batch['label'].to(rank)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through wav2vec2 for feature extraction
            inputs = Wav2Vec2_tokenizer(waveforms.squeeze().cpu().numpy(), sampling_rate=batch['sample_rate'], return_tensors="pt", padding="longest").to(rank)
            features = Wav2Vec2_model(input_values=inputs['input_values']).last_hidden_state
            # print(f'type {type(features)}  with size {features.size()} , features= {features}')

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(rank)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = PS_Model(features,lengths)


            # Calculate loss
            loss = criterion(outputs, labels)  
            epoch_loss += loss.item()


            # Backward pass and optimization
            loss.backward()
            optimizer.step()


            with torch.no_grad():  # No need to compute gradients for EER calculation

                # Calculate utterance predictions
                utterance_predictions.extend(get_uttEER_by_seg(outputs,labels))
                # utterance_pooling_predictions.extend(utterance_pooling_scores)
                # print(f"utterance_pooling_predictions shape= {utterance_pooling_predictions.shape()},\n utterance_pooling_predictions= {utterance_pooling_predictions}")
                # print(f" utterance_pooling_predictions= {utterance_pooling_predictions}")
                segment_predictions.extend(outputs)
                segment_labels.extend(labels)

                # Accumulate files names
                if epoch == 0:
                    batch_file_names = batch['file_name']
                    files_names.extend(batch_file_names)


            # Print batch training progress
            # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch Loss: {loss.item()}, Batch Segment EER: {segment_eer:.4f}, Batch Segment EER Threshold: {segment_eer_threshold:.4f}')



        # Save checkpoint
        if NUM_EPOCHS>=save_interval and (epoch + 1) % (NUM_EPOCHS//save_interval) == 0:
            # Generate a unique filename based on hyperparameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
                        
            save_checkpoint(PS_Model, optimizer, epoch + 1,os.path.join(model_save_path,model_filename))


        # Get Average Utterance EER for the epoch
        if epoch ==0: utterance_labels =[PartialSpoof_LA_cm_train_trl_dict[file_name] for file_name in files_names]
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # utterance_pooling_predictions = torch.cat(utterance_pooling_predictions, dim=0)
        # utterance_pooling_eer, utterance_pooling_eer_threshold = compute_eer(utterance_pooling_predictions,torch.tensor(utterance_labels))

        # Calculate Training segment EER
        segment_predictions=torch.cat(segment_predictions, dim=0)
        segment_labels=torch.cat(segment_labels, dim=0)
        segment_eer, segment_eer_threshold = compute_eer(segment_predictions,segment_labels)

        # Average  loss for the epoch
        epoch_loss /= len(train_loader)

        # Print epoch training progress
        # print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete. Average Loss /epoch : {epoch_loss:.4f},\n'
        #        f'Average Segment EER: {segment_eer:.4f}, Average Segment EER Threshold: {segment_eer_threshold:.4f},\n'
        #        f'Average Utterance EER: {utterance_eer:.4f}, Average Utterance EER Threshold: {utterance_eer_threshold:.4f}')


        training_segment_eer_per_epoch.append(segment_eer)

        BASE_DIR = os.getcwd()
        # Define development files and labels
        dev_files_path=os.path.join(BASE_DIR,'database/dev/con_wav')
        # dev_files_path=os.path.join(BASE_DIR,'database/mini_database/dev')
        dev_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/dev_seglab_0.64.npy')
        dev_seglab_64_dict = np.load(dev_seglab_64_path, allow_pickle=True).item()

        dev_metrics_dict=dev_model(rank,world_size, PS_Model,dev_files_path, dev_seglab_64_dict, Wav2Vec2_tokenizer,Wav2Vec2_model, BATCH_SIZE,DEVICE=DEVICE)
        dev_segment_eer_per_epoch.append(dev_metrics_dict['segment_eer'])

        wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
            'training_segment_eer_epoch': segment_eer, 
            'training_segment_eer_threshold_epoch': segment_eer_threshold,
            'training_utterance_eer_epoch': utterance_eer,
            'training_utterance_eer_threshold_epoch': utterance_eer_threshold, 
            # 'training_utterance_pooling_eer_epoch': utterance_pooling_eer,
            # 'training_utterance_pooling_eer_threshold_epoch': utterance_pooling_eer_threshold, 
            'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
            'validation_segment_eer_epoch': dev_metrics_dict['segment_eer'], 
            'validation_segment_eer_threshold_epoch': dev_metrics_dict['segment_eer_threshold'],
            'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
            'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold']                      
            })


        # Early stopping check
        early_stopping(dev_metrics_dict['epoch_loss'], PS_Model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step()

    # plot training EER per epoch
    plot_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
    # plot_train_dev_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch, dev_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
    # # plot Vali EER per epoch
    # plot_eer_per_epoch(NUM_EPOCHS, training_eer_per_epoch)


    # Generate a unique filename based on hyperparameters
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_filename = f"model_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
    
    # Save the trained model
    model_save_path=os.path.join(model_save_path,model_filename)
    # torch.save(PS_Model.state_dict(), model_save_path)
    save_checkpoint(PS_Model, optimizer,NUM_EPOCHS,model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save segment_predictions, segment_labels, utterance_predictions, utterance_labels
    torch.save(segment_predictions,os.path.join(os.getcwd(),f'outputs/segment_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(utterance_predictions,os.path.join(os.getcwd(),f'outputs/utterance_predictions_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(segment_labels,os.path.join(os.getcwd(),f'outputs/segment_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))
    torch.save(torch.tensor(utterance_labels),os.path.join(os.getcwd(),f'outputs/utterance_labels_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pt'))

    # Save last metrics
    training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,segment_eer,segment_eer_threshold,epoch_loss)
    training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs/{training_metrics_dict_filename}')
    save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

    # Sync the loss across all processes (optional)
    dist.barrier()
    cleanup()  # Clean up distributed environment
    
    if DEVICE=='cuda': torch.cuda.empty_cache()
    wandb.finish()
    print("Training complete!")



def train():
    # Initialize W&B
    wandb.init(project='partial_spoof_trial_2')

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
    train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()

    # Load the tokenizer and model from the local directory
    # Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    # Wav2Vec2_model.eval()


    # Get the rank and world size from the environment variables
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    # Call train_model with parameters from W&B sweep
    train_model(
        rank=rank,
        world_size=world_size,
        train_directory=train_files_path,
        train_labels_dict=train_seglab_64_dict,
        BATCH_SIZE=config.BATCH_SIZE,
        NUM_EPOCHS=config.NUM_EPOCHS,
        LEARNING_RATE=config.LEARNING_RATE,
        DEVICE=DEVICE,
        save_interval=10
    )




if __name__ == "__main__":

    # Device configuration
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    BASE_DIR = os.getcwd()

    # Define training files and labels
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train')
    # train_files_path=os.path.join(BASE_DIR,'database/mini_database/train2')
    train_files_path=os.path.join(BASE_DIR,'database/mini_database/train3')
    train_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


    # Load the tokenizer and model from the local directory
    # Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    # Wav2Vec2_model.eval()


    # Record the start time
    start_time = datetime.now()
    # train model
    train_model(train_files_path, train_seglab_64_dict,  BATCH_SIZE=16,NUM_EPOCHS=5,DEVICE=DEVICE)

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


    # file_name='CON_T_0000000'
    # print(train_seglab_64_dict[file_name])
    # print(type(train_seglab_64_dict[file_name]))
