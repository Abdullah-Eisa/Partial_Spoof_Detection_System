
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from datetime import datetime

from utils import *
from preprocess import *
from model import *

# ===========================================================================================================================
def inference_helper(model,test_data_loader, test_labels_dict,criterion,DEVICE='cpu'):
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
    dropout_prob=0
    # c=0
    with torch.no_grad():
        for batch in tqdm(test_data_loader, desc="Test Batches", leave=False):
            # if c>8:
            #     break
            # else:
            #     c+=1
            fbank = batch['fbank'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Pass features to model and get predictions
            outputs = forward_pass(model,fbank)

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in test loop loss") 
                continue
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels =torch.tensor([test_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_data_loader)


    # Print epoch testing results
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)





def inference(input_fdim,input_tdim, imagenet_pretrain, audioset_pretrain, model_size,
    eval_data_path=os.path.join(os.getcwd(),'database/eval/con_wav'),
    eval_labels_path = os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json'),
    AST_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'),
    BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE='cpu'):

    print("infer_model is working ... ")
    print(f'inference device: {DEVICE}')
    # Define your paths and other fixed arguments
    # BASE_DIR = os.getcwd()

    # Define training files and labels
    eval_labels_dict= load_json_dictionary(eval_labels_path)
    pin_memory= True if DEVICE=='cuda' else False   # Enable page-locked memory for faster data transfer to GPU
    eval_data_loader = initialize_data_loader(eval_data_path, eval_labels_path,BATCH_SIZE,False, num_workers, prefetch_factor,pin_memory)

    # Initialize Binary Spoofing Classification Model
    AST_model = ASTModel(input_fdim=input_fdim, input_tdim=input_tdim, 
                         imagenet_pretrain=imagenet_pretrain, 
                         audioset_pretrain=audioset_pretrain, 
                         model_size=model_size).to(DEVICE)
    # PS_Model_name='model_epochs60_batch8_lr0.005_20241226_214707.pth'
    # PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/{PS_Model_name}')

    # PS_Model,_,_=load_checkpoint(PS_Model, optimizer, path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs30_batch8_lr0.005_20241216_013405.pth'))
    checkpoint = torch.load(AST_Model_path)
    AST_model.load_state_dict(checkpoint['model_state_dict'])
    AST_model.eval()  # Set the model to evaluation mode

    criterion = initialize_loss_function().to(DEVICE)

    inference_helper(
        model=AST_model,
        test_data_loader=eval_data_loader, 
        test_labels_dict=eval_labels_dict,
        criterion=criterion,
        DEVICE=DEVICE)

    if DEVICE=='cuda': torch.cuda.empty_cache()


# ===========================================================================================================================
def dev_one_epoch(model, dev_data_loader, dev_labels_dict,criterion,dropout_prob=0,DEVICE='cpu'):
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
    c=0
    with torch.no_grad():
        for batch in tqdm(dev_data_loader, desc="Dev Batches", leave=False):
            if c>2:
                break
            else:
                c+=1
            fbank = batch['fbank'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            labels = labels.unsqueeze(1).float()   # Converts labels from shape [batch_size] to [batch_size, 1]

            # Pass features to model and get predictions
            # outputs = PS_Model(features,lengths,dropout_prob)
            outputs = forward_pass(model,fbank)
            print(f"outputs: {outputs}")

            # Calculate loss
            loss = criterion(outputs, labels) 
            if torch.isnan(loss).any(): 
                print(f"NaN detected in validation loop loss") 
                continue
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(outputs)
                # Accumulate files names
                files_names.extend(batch['file_name'])


        # Get Average Utterance EER for the epoch
        utterance_labels =torch.tensor([dev_labels_dict[file_name] for file_name in files_names])
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_metrics(utterance_predictions,utterance_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_data_loader)


    # Print epoch dev progress
    # print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
    #            f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
    #            f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_loss)



if __name__ == "__main__":
    print("SA, inference.py file !")
    # Record the start time
    start_time = datetime.now()

    inference(AST_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'))

    # inference(eval_data_path=os.path.join(os.getcwd(),'database/eval/con_wav'),
    #     eval_labels_path = os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json'),
    #     ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt'),
    #     PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'),
    #     feature_dim=768,num_heads=8,hidden_dim=128,max_dropout=0,depthwise_conv_kernel_size=31,
    #     conformer_layers=1,max_pooling_factor=3,
    #     BATCH_SIZE=16, num_workers=0, prefetch_factor=None, DEVICE='cuda')

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
