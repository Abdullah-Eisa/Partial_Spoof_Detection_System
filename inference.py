
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime

from utils import *
from preprocess import *
from model import *

# ===========================================================================================================================
# def inference_helper(model, feature_extractor,criterion,
#                   test_data_loader, test_labels_dict,DEVICE='cpu'):
def inference_helper(model, feature_extractor,criterion,
                  test_data_loader, DEVICE='cpu'):
    """Evaluate the model on the test set"""

    # testing phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names=[]

    epoch_loss = 0
    utterance_predictions=[]
    utterance_labels=[]
    dropout_prob=0
    # c=0
    nan_count=0
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in test loop loss") 
                nan_count+=torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in train_one_epoch batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                utterance_predictions.extend(outputs)
                utterance_labels.extend(labels)
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        # utterance_labels =torch.tensor([test_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_data_loader)


    # Print epoch testing results
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')
    print("===================================================")
    print(f'In Test loop, Total loss NAN count: {nan_count}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)





def inference(eval_data_path=os.path.join(os.getcwd(),'database/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac'),
    eval_labels_path = os.path.join(os.getcwd(),'database/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trn.txt'),
    ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt'),
    PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'),
    feature_dim=768,num_heads=8,hidden_dim=128,max_dropout=0,depthwise_conv_kernel_size=31,
    conformer_layers=1,max_pooling_factor=3,
    BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE='cpu'):

    print("infer_model is working ... ")
    # Get Device
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'inference device: {DEVICE}')

    # Define your paths and other fixed arguments
    BASE_DIR = os.getcwd()

    # Define training files and labels
    # eval_labels_dict= load_json_dictionary(eval_labels_path)
    # eval_labels_dict= load_labels_txt2dict(eval_labels_path)
    pin_memory= True if DEVICE=='cuda' else False   # Enable page-locked memory for faster data transfer to GPU
    eval_data_loader = initialize_data_loader(eval_data_path, eval_labels_path,BATCH_SIZE,False, num_workers, prefetch_factor,pin_memory)


    # Load feature extractor
    # ssl_ckpt_path = os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')
    # ssl_ckpt_name='w2v_large_lv_fsh_swbd_cv_20241223_152156.pt'
    # ssl_ckpt_path = os.path.join(os.getcwd(), f'models/back_end_models/{ssl_ckpt_name}')
    feature_extractor = torch.hub.load('s3prl/s3prl', 'wav2vec2', model_path=ssl_ckpt_path).to(DEVICE)
    feature_extractor.eval()

    # Initialize Binary Spoofing Classification Model
    PS_Model = BinarySpoofingClassificationModel(feature_dim, num_heads, hidden_dim, max_dropout, depthwise_conv_kernel_size, conformer_layers, max_pooling_factor).to(DEVICE)
    # PS_Model_name='model_epochs60_batch8_lr0.005_20241226_214707.pth'
    # PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/{PS_Model_name}')

    # PS_Model,_,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs30_batch8_lr0.005_20241216_013405.pth'))
    checkpoint = torch.load(PS_Model_path)
    PS_Model.load_state_dict(checkpoint['model_state_dict'])
    # PS_Model.load_state_dict(checkpoint)
    PS_Model.eval()  # Set the model to evaluation mode


    criterion = initialize_loss_function().to(DEVICE)


    inference_helper(
        model=PS_Model,
        feature_extractor=feature_extractor,
        criterion=criterion,
        test_data_loader=eval_data_loader, 
        DEVICE=DEVICE)

    if DEVICE=='cuda': torch.cuda.empty_cache()


# ===========================================================================================================================
# def dev_one_epoch(model, feature_extractor,criterion,
#                   dev_data_loader, dev_labels_dict,dropout_prob=0,DEVICE='cpu'):
def dev_one_epoch(model, feature_extractor,criterion,
                  dev_data_loader,dropout_prob=0,DEVICE='cpu'):
    """Evaluate the model on the development set"""

    # Validation phase
    model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")

    # Initialize variables
    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    utterance_predictions=[]
    utterance_labels=[]
    c=0
    nan_count=0
    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc="Dev Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Forward pass through wav2vec2 for feature extraction
            features = feature_extractor(waveforms)['hidden_states'][-1] 
             # print(f'type {type(features)}  with size {features.size()} , features= {features}')

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            # outputs = PS_Model(features,lengths,dropout_prob)
            outputs = forward_pass(model, features, lengths, dropout_prob)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in loss during development loop")
                # c+=1
                nan_count+=torch.isnan(loss).sum().item()
                print(f"loss value: {loss.item()}")
                print(f"batch['file_name']: {batch['file_name']}")
                print(f"in dev_one_epoch batch, nan_count: {nan_count}")
                continue

            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                utterance_labels.extend(labels)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        # utterance_labels =torch.tensor([dev_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_labels = torch.cat(utterance_labels)
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_data_loader)


    # Print epoch dev progress
    # print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
    #            f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
    #            f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')
    print("===================================================")
    print(f'In Dev loop, Total loss NAN count: {nan_count}')
    
    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss), nan_count



if __name__ == "__main__":
    print("SA, inference.py file !")
    # Record the start time
    start_time = datetime.now()

    inference(PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'))

    # inference(eval_data_path=os.path.join(os.getcwd(),'database/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac'),
    #     eval_labels_path = os.path.join(os.getcwd(),'database/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trn.txt'),
    #     ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt'),
    #     PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'),
    #     feature_dim=768,num_heads=8,hidden_dim=128,max_dropout=0,depthwise_conv_kernel_size=31,
    #     conformer_layers=1,max_pooling_factor=3,
    #     BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE='cpu')

    # Record the end time
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"Total time: {total_time}")

    # Extract hours, minutes, and seconds
    total_seconds = total_time.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Print training time in hours, minutes, and seconds
    print(f"Total time: {hours} hours, {minutes} minutes, {seconds} seconds")
