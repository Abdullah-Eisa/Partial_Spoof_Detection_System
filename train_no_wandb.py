# Check --module-config config_ps.config_test_on_eval , for the following input:

# mostly like PS , I will need
# trn_set_name, \
# trn_lst,
# trn_input_dirs, \
# input_exts, \
# input_dims, \
# input_reso, \
# input_norm, \
# output_dirs, \
# output_exts, \
# output_dims, \
# output_reso, \
# output_norm, \
# params 
# truncate_seq  
# min_seq_len 
# save_mean_std
# wav_samp_rate 

### check possible num_workers & prefetch_factor other than  num_workers=0, prefetch_factor=None for parallelized training


# #stage 1:
# if [ $stage -le 1 ]; then
#     python main.py --module-model model --model-forward-with-file-name --seed 1 \
# 	--ssl-finetune \
# 	--multi-scale-active utt 64 32 16 8 4 2 \
# 	--num-workers 4 --epochs 5000 --no-best-epochs 50 --batch-size 8 --not-save-each-epoch\
#        	--sampler block_shuffle_by_length --lr-decay-factor 0.5 --lr-scheduler-type 1 --lr 0.00001 \
# 	--module-config config_ps.config_test_on_eval \
# 	--temp-flag ${CON_PATH}/segment_labels/train_seglab_0.01.npy \
# 	--temp-flag-dev ${CON_PATH}/segment_labels/dev_seglab_0.01.npy --dev-flag >  ${OUTPUT_DIR}/log_train 2> ${OUTPUT_DIR}/log_err
# fi





#in the first 1600 training examples:  The maximum size in the second dimension of the tensors listed is 393.




# wav_samp_rate = 16000
# truncate_seq = None

# batch_size=8
# epochs=5000
# learning_rate=0.00001




import os
from tqdm import tqdm  # Correctly import tqdm
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# from transformers import Wav2Vec2Processor, 
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model

# import wandb

from utils import *
from model import *
from inference import dev_model


# ... (your training and inference functions)

def train_model(train_directory, train_labels_dict, 
                tokenizer,feature_extractor,
                BATCH_SIZE=32, NUM_EPOCHS=1,LEARNING_RATE=0.001,
                model_save_path=os.path.join(os.getcwd(),'models\\back_end_models'),
                DEVICE='cpu',save_interval=3):

    # Initialize W&B
    # wandb.init(project='partial_spoof_trial_0')

    # Ensure the model save path exists
    os.makedirs(model_save_path, exist_ok=True)
    # Load utterance labels
    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_train_trl_dict_path = os.path.join(BASE_DIR,'database\\utterance_labels\\PartialSpoof_LA_cm_train_trl.json')
    PartialSpoof_LA_cm_train_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_train_trl_dict_path)

    # Initialize the model, loss function, and optimizer
    hidd_dims ={'wav2vec2':768, 'wav2vec2_large':1024}
    PS_Model = MyModel(d_model=hidd_dims['wav2vec2']).to(DEVICE)  # Move model to the configured device
    # criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss with Logits for multi-label classification
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for multi-label classification
    optimizer = optim.Adam(PS_Model.parameters(), lr=LEARNING_RATE)
    
    # Get the data loader
    train_loader = get_audio_data_loaders(train_directory, train_labels_dict, tokenizer,feature_extractor, batch_size=BATCH_SIZE, shuffle=True)

    PS_Model.train()  # Set the model to training mode
    

    files_names=[]
    training_segment_eer_per_epoch=[]
    dev_segment_eer_per_epoch=[]

    for epoch in tqdm(range(NUM_EPOCHS), desc="Epochs"):
        PS_Model.train()  # Set the model to training mode

        epoch_loss = 0
        epoch_segment_eer = 0
        epoch_segment_eer_threshold = 0
        utterance_eer, utterance_eer_threshold=0,0
        utterance_predictions=[]
        # utterance_labels=[]

        for batch in tqdm(train_loader, desc="Train Batches", leave=False):
            features = batch['features'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Pass features to model and get predictions
            outputs = PS_Model(features)


            # Calculate loss
            loss = criterion(outputs, labels.float())  # Convert labels to float for BCELoss
            epoch_loss += loss.item()


            # Backward pass and optimization
            loss.backward()
            optimizer.step()


            with torch.no_grad():  # No need to compute gradients for EER calculation

                # Calculate utterance predictions
                # utterance_predictions.extend(torch.max(outputs, dim=1, keepdim=True)[0])
                utterance_predictions.extend(torch.max(outputs, dim=1, keepdim=True).values)
                # utterance_predictions.extend(torch.min(outputs, dim=1, keepdim=True).values)

                # Calculate segment EER
                batch_segment_eer, batch_segment_eer_threshold = compute_eer(outputs, labels)
                epoch_segment_eer += batch_segment_eer
                epoch_segment_eer_threshold += batch_segment_eer_threshold


                # Accumulate files names
                if epoch == 0:
                    batch_file_names = batch['file_name']
                    files_names.extend(batch_file_names)


            # Print batch training progress
            # print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Batch Loss: {loss.item()}, Batch Segment EER: {segment_eer:.4f}, Batch Segment EER Threshold: {segment_eer_threshold:.4f}')

        # Save checkpoint
        if (epoch + 1) % (NUM_EPOCHS//save_interval) == 0:
            # Generate a unique filename based on hyperparameters
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_filename = f"model_epochs{epoch + 1}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.pth"
                        
            save_checkpoint(PS_Model, optimizer, epoch + 1,os.path.join(model_save_path,model_filename))

        # Get Average Utterance EER for the epoch
        if epoch ==0: utterance_labels =[PartialSpoof_LA_cm_train_trl_dict[file_name] for file_name in files_names]
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # Average Segment EER and loss for the epoch
        epoch_loss /= len(train_loader)
        epoch_segment_eer /= len(train_loader)
        epoch_segment_eer_threshold /= len(train_loader)

        # Print epoch training progress
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}] Complete. Average Loss /epoch : {epoch_loss:.4f},\n'
               f'Average Segment EER: {epoch_segment_eer:.4f}, Average Segment EER Threshold: {epoch_segment_eer_threshold:.4f},\n'
               f'Average Utterance EER: {utterance_eer:.4f}, Average Utterance EER Threshold: {utterance_eer_threshold:.4f}')


        training_segment_eer_per_epoch.append(epoch_segment_eer)

        BASE_DIR = os.getcwd()
        # Define development files and labels
        # dev_files_path=os.path.join(BASE_DIR,'database\\dev\\con_wav')
        dev_files_path=os.path.join(BASE_DIR,'database\\mini_database\\dev')
        dev_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\dev_seglab_0.64.npy')
        dev_seglab_64_dict = np.load(dev_seglab_64_path, allow_pickle=True).item()

        dev_metrics_dict=dev_model( PS_Model,dev_files_path, dev_seglab_64_dict, tokenizer,feature_extractor, BATCH_SIZE,DEVICE=DEVICE)
        dev_segment_eer_per_epoch.append(dev_metrics_dict['segment_eer'])

        # wandb.log({'epoch': epoch+1,'training_loss_epoch': epoch_loss,
        #     'training_segment_eer_epoch': epoch_segment_eer, 
        #     'training_segment_eer_threshold_epoch': epoch_segment_eer_threshold,
        #     'training_utterance_eer_epoch': utterance_eer,
        #     'training_utterance_eer_threshold_epoch': utterance_eer_threshold, 
        #     'validation_loss_epoch': dev_metrics_dict['epoch_loss'],
        #     'validation_segment_eer_epoch': dev_metrics_dict['segment_eer'], 
        #     'validation_segment_eer_threshold_epoch': dev_metrics_dict['segment_eer_threshold'],
        #     'validation_utterance_eer_epoch': dev_metrics_dict['utterance_eer'],
        #     'validation_utterance_eer_threshold_epoch': dev_metrics_dict['utterance_eer_threshold']                      
        #     })

    # plot training EER per epoch
    plot_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
    plot_train_dev_eer_per_epoch(NUM_EPOCHS, training_segment_eer_per_epoch, dev_segment_eer_per_epoch,os.path.join(os.getcwd(),'outputs'))
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


    # Save metrics
    training_metrics_dict=create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_segment_eer,epoch_segment_eer_threshold,epoch_loss)
    training_metrics_dict_filename = f"metrics_dict_epochs{NUM_EPOCHS}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.json"
    training_metrics_dict_save_path=os.path.join(os.getcwd(),f'outputs\\{training_metrics_dict_filename}')
    save_json_dictionary(training_metrics_dict_save_path,training_metrics_dict)

    if DEVICE=='cuda': torch.cuda.empty_cache()
    # wandb.finish()
    print("Training complete!")





if __name__ == "__main__":

    # Device configuration
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    BASE_DIR = os.getcwd()

    # Define training files and labels
    # train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train')
    # train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train2')
    train_files_path=os.path.join(BASE_DIR,'database\\mini_database\\train3')
    train_seglab_64_path=os.path.join(BASE_DIR,'database\\segment_labels\\train_seglab_0.64.npy')
    train_seglab_64_dict = np.load(train_seglab_64_path, allow_pickle=True).item()


    # Load the tokenizer and model from the local directory
    Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
    Wav2Vec2_model.eval()


    # Record the start time
    start_time = datetime.now()
    # train model
    train_model(train_files_path, train_seglab_64_dict, Wav2Vec2_tokenizer,Wav2Vec2_model, BATCH_SIZE=16,NUM_EPOCHS=3,DEVICE=DEVICE)

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
