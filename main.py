
import wandb
import os
from datetime import datetime
import torch

from train import train
from inference import inference


def main():
    """ main(): the default wrapper for training and inference process
    """
    # wandb_key="Get the key here"
    wandb_api_key="c1fc533d0bafe63c83a9110c6daef36b76f77de1"
    # os.system(f"echo {wandb_key}")
    wandb.login(key=wandb_api_key,relogin=True,force=True)

    # Choose the dataset to train on
    dataset_namses_set= ['RFP_Dataset','PartialSpoof_Dataset','ASVspoof2019_Dataset']
    dataset_name=dataset_namses_set[1]

    sweep_config = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'validation_utterance_eer_epoch'
            },
        'parameters': 
        {   
            # Hyperparameters to tune
            # 'NUM_EPOCHS': {'values': [5, 7]},
            # 'BATCH_SIZE': {'values': [16,32]},
            'NUM_EPOCHS': {'values': [19]},
            'LEARNING_RATE': {'values': [0.00075]},
            'BATCH_SIZE': {'values': [8]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

            # Add additional parameters here
            'dataset_name': {'value': dataset_name},  # Changed to lowercase
            'train_data_path': {'value': os.path.join(os.getcwd(),'database/PartialSpoof/database/train/con_wav')},  # Changed to lowercase
            'train_labels_path': {'value': os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_train_trl.json')},  # Changed to lowercase
            'dev_data_path': {'value': os.path.join(os.getcwd(), 'database/PartialSpoof/database/dev/con_wav')},  # Changed to lowercase
            'dev_labels_path': {'value': os.path.join(os.getcwd(), 'database/utterance_labels/PartialSpoof_LA_cm_dev_trl.json')},  # Changed to lowercase
            'ssl_ckpt_path': {'value': os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt')},  # Added path to SSL checkpoint
            'apply_transform': {'value': False},  # Added boolean parameter
            'save_feature_extractor': {'value': False},  # Added boolean parameter
            'feature_dim': {'value': 768},  # Added fixed parameter
            'num_heads': {'value': 8},  # Added fixed parameter
            'hidden_dim': {'value': 128},  # Added fixed parameter
            'max_dropout': {'value': 0.35},  # Added fixed parameter
            'depthwise_conv_kernel_size': {'value': 31},  # Added fixed parameter
            'conformer_layers': {'value': 1},  # Added fixed parameter
            'max_pooling_factor': {'value': 3},  # Added fixed parameter
            'num_workers': {'value': 8},  # Added fixed parameter
            'prefetch_factor': {'value': 2},  # Added fixed parameter
            'pin_memory': {'value': True},  # Added fixed parameter
            'monitor_dev_epoch': {'value': 0},  # Added fixed parameter
            'save_interval': {'value': 5},  # Added fixed parameter
            'model_save_path': {'value': os.path.join(os.getcwd(),'models/back_end_models')},  # Added model save path
            'patience': {'value': 15},  # Added patience value
            'max_grad_norm': {'value': 1.0},  # Added gradient clipping value
            'gamma': {'value': 0.9},  # Added gamma value
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'cpu'},  # Added device
        

        }
    }


    project_name=f'{dataset_name}_Wav2Vec2_Conformer_binary_classifier'

    sweep_id = wandb.sweep(sweep=sweep_config,project=project_name)
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=1)




if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    main()

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





