
import os
from tqdm import tqdm  # Correctly import tqdm
# from transformers import Wav2Vec2Processor, 
from transformers import Wav2Vec2Tokenizer, Wav2Vec2Model
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim


from gmlp import GMLPBlock
from utils import *
from model import *



def dev_model( PS_Model,dev_directory, labels_dict, tokenizer,feature_extractor, BATCH_SIZE=32,epoch=0,DEVICE='cpu'):

    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_dev_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json')
    PartialSpoof_LA_cm_dev_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_dev_trl_dict_path)

    # Get the data loader

    # dev_loader = get_audio_data_loaders(dev_directory, labels_dict, tokenizer,feature_extractor, batch_size=BATCH_SIZE, shuffle=True)
    dev_loader = get_raw_labeled_audio_data_loaders(dev_directory, labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=8, prefetch_factor=2)
    # loader_iter = iter(data_loader) # preloading starts here

    # with the default prefetch_factor of 2, 2*num_workers=16 batches will be preloaded
    # the max index printed by __getitem__ is thus 31 (16*batch_size=32 samples loaded)

    # data = next(loader_iter) # this will consume a batch and preload the next one from a single worker to fill the queue
    # batch_size=2 new samples should be loaded
    
    # Validation phase
    PS_Model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")



    # Calculate loss
    # loss = criterion(outputs, labels.float())  # Convert labels to float for BCELoss
    criterion = CustomLoss().to(DEVICE)

    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    segment_eer, segment_eer_threshold=0,0
    utterance_predictions=[]
    segment_predictions=[]
    segment_labels=[]
    c=0
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev Batches", leave=False):
        # for i in range(len(data_loader)):
        #     data = next(loader_iter)
        #     waveforms = data['waveform'].to(DEVICE)
        #     labels = data['label'].to(DEVICE)
            # if c>8:
            #     break
            # else:
            #     c+=1
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Forward pass through wav2vec2 for feature extraction
            inputs = tokenizer(waveforms.squeeze().cpu().numpy(), sampling_rate=batch['sample_rate'], return_tensors="pt", padding="longest").to(DEVICE)
            features = feature_extractor(input_values=inputs['input_values']).last_hidden_state
            # print(f'type {type(features)}  with size {features.size()} , features= {features}')

            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = PS_Model(features,lengths)

            # Calculate loss
            loss = criterion(outputs, labels) 
            epoch_loss += loss.item()

            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(get_uttEER_by_seg(outputs,labels))

                segment_predictions.extend(outputs)
                segment_labels.extend(labels)


                # Accumulate files names
                if epoch == 0:
                    batch_file_names = batch['file_name']
                    files_names.extend(batch_file_names)


        # Get Average Utterance EER for the epoch
        if epoch ==0: utterance_labels =[PartialSpoof_LA_cm_dev_trl_dict[file_name] for file_name in files_names]
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # Calculate Training segment EER
        segment_predictions=torch.cat(segment_predictions, dim=0)
        segment_labels=torch.cat(segment_labels, dim=0)
        segment_eer, segment_eer_threshold = compute_eer(segment_predictions,segment_labels)

        # Average loss for the epoch
        epoch_loss /= len(dev_loader)


    # Print epoch dev progress
    print(f'Epoch [{epoch + 1}] Complete. Validation Loss: {epoch_loss:.4f},\n'
               f'Average Validation Segment EER: {segment_eer:.4f}, Average Validation Segment EER Threshold: {segment_eer_threshold:.4f},\n'
               f'Average Validation Utterance EER: {utterance_eer:.4f}, Average Validation Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,segment_eer,segment_eer_threshold,epoch_loss)
    




def infer_model(model_path,test_directory, test_labels_dict, tokenizer, feature_extractor, BATCH_SIZE=32,DEVICE='cpu'):
    # Initialize the model
    PS_Model = MyModel().to(DEVICE)  # Initialize the model and move to the configured device
    # model_path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs1_batch16_lr0.001_20240911_234211.pth')
    # PS_Model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # Load the trained model weights to the correct device
    _=load_checkpoint(PS_Model,optim.Adam(PS_Model.parameters()), path=model_path) 
    PS_Model.eval()  # Set the model to evaluation mode

    # Wrap the model with DataParallel
    if torch.cuda.device_count() > 1:
        PS_Model = nn.DataParallel(PS_Model).to(DEVICE)
        print("Parallelizing model on ", torch.cuda.device_count(), "GPUs!")


    # Get the test data loader
    # test_loader = get_audio_data_loaders(test_directory, test_labels_dict, tokenizer, feature_extractor, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = get_raw_labeled_audio_data_loaders(test_directory, test_labels_dict,batch_size=BATCH_SIZE, shuffle=True, num_workers=6, prefetch_factor=2)
    # loader_iter = iter(data_loader) # preloading starts here

    # with the default prefetch_factor of 2, 2*num_workers=16 batches will be preloaded
    # the max index printed by __getitem__ is thus 31 (16*batch_size=32 samples loaded)

    # data = next(loader_iter) # this will consume a batch and preload the next one from a single worker to fill the queue
    # batch_size=2 new samples should be loaded

    # Get Utterance lables dictionary    
    BASE_DIR = os.getcwd()
    PartialSpoof_LA_cm_eval_trl_dict_path = os.path.join(BASE_DIR,'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
    PartialSpoof_LA_cm_eval_trl_dict= load_json_dictionary(PartialSpoof_LA_cm_eval_trl_dict_path)

    # Calculate loss
    # loss = criterion(outputs, labels.float())  # Convert labels to float for BCELoss
    criterion = CustomLoss().to(DEVICE)

    files_names=[]

    epoch_loss = 0
    utterance_eer, utterance_eer_threshold=0,0
    segment_eer, segment_eer_threshold=0,0
    utterance_predictions=[]
    segment_predictions=[]
    segment_labels=[]

    # all_predictions = []
    # all_labels = []

    with torch.no_grad():  # Disable gradient calculation during inference
        for batch in tqdm(test_loader, desc="Test Batches", leave=False):
        # for i in range(len(data_loader)):
        #     data = next(loader_iter)
        #     waveforms = data['waveform'].to(DEVICE)
        #     labels = data['label'].to(DEVICE)
            waveforms = batch['waveform'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward pass through wav2vec2 for feature extraction
            inputs = tokenizer(waveforms.squeeze().cpu().numpy(), sampling_rate=batch['sample_rate'], return_tensors="pt", padding="longest").to(DEVICE)
            features = feature_extractor(input_values=inputs['input_values']).last_hidden_state
            # print(f'type {type(features)}  with size {features.size()} , features= {features}')
            
            # lengths should be the number of non-padded frames in each sequence
            lengths = torch.full((features.size(0),), features.size(1), dtype=torch.int16).to(DEVICE)  # (batch_size,)

            # Pass features to model and get predictions
            outputs = PS_Model(features,lengths)

            loss = criterion(outputs, labels) 
            epoch_loss += loss.item()


            with torch.no_grad():
                # Calculate utterance predictions
                utterance_predictions.extend(get_uttEER_by_seg(outputs,labels))

                segment_predictions.extend(outputs)
                segment_labels.extend(labels)

                # Accumulate files names
                batch_file_names = batch['file_name']
                files_names.extend(batch_file_names)


        # Get Average Utterance EER for the epoch
        utterance_labels =[PartialSpoof_LA_cm_eval_trl_dict[file_name] for file_name in files_names]
        # print(f'epoch {epoch} , utterance_labels: {utterance_labels}')
        utterance_predictions = torch.cat(utterance_predictions)
        utterance_eer, utterance_eer_threshold = compute_eer(utterance_predictions,torch.tensor(utterance_labels))

        # Calculate  segment EER
        segment_predictions=torch.cat(segment_predictions, dim=0)
        segment_labels=torch.cat(segment_labels, dim=0)
        segment_eer, segment_eer_threshold = compute_eer(segment_predictions,segment_labels)

        # Average loss for the epoch
        epoch_loss /= len(test_loader)


    # Print epoch dev progress
    print(f'Testing/Inference Complete. Test Loss: {epoch_loss:.4f},\n'
               f'Average Test Segment EER: {segment_eer:.4f}, Average Test Segment EER Threshold: {segment_eer_threshold:.4f},\n'
               f'Average Test Utterance EER: {utterance_eer:.4f}, Average Test Utterance EER Threshold: {utterance_eer_threshold:.4f}')

    return create_metrics_dict(utterance_eer,utterance_eer_threshold,epoch_segment_eer,epoch_segment_eer_threshold,epoch_loss)




#in the first 1600 training examples:  The maximum size in the second dimension of the tensors listed is 393.

if __name__ == "__main__":

    # Device configuration
    use_cuda= True
    use_cuda =  use_cuda and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    print(f'device: {DEVICE}')

    BASE_DIR = os.getcwd()

    # Define testing files and labels
    test_files_path=os.path.join(BASE_DIR,'database/mini_database/eval')
    test_seglab_64_path=os.path.join(BASE_DIR,'database/segment_labels/eval_seglab_0.64.npy')
    test_seglab_64_dict = np.load(test_seglab_64_path, allow_pickle=True).item()


    # Load the tokenizer and model from the local directory
    Wav2Vec2_tokenizer = Wav2Vec2Tokenizer.from_pretrained("models/local_wav2vec2_tokenizer")
    # Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model")
    Wav2Vec2_model = Wav2Vec2Model.from_pretrained("models/local_wav2vec2_model").to(DEVICE)
    Wav2Vec2_model.eval()

    # Backend model path
    # model_path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs1_batch16_lr0.001_20240911_234211.pth')
    model_path=os.path.join(os.getcwd(),'models/back_end_models/model_epochs3_batch16_lr0.001_20240930_110553.pth')

    # inference
    inference_metrics_dict=infer_model(model_path,test_files_path, test_seglab_64_dict, Wav2Vec2_tokenizer,Wav2Vec2_model, BATCH_SIZE=16,DEVICE=DEVICE)


    # file_name='CON_T_0000000'
    # print(train_seglab_64_dict[file_name])
    # print(type(train_seglab_64_dict[file_name]))