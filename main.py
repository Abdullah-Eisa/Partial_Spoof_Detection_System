
import wandb
import os
from datetime import datetime

from train import train
from inference import inference


def main():
    """ main(): the default wrapper for training and inference process
    """

    # wandb_key="Get the key here"
    wandb_api_key="c1fc533d0bafe63c83a9110c6daef36b76f77de1"
    # os.system(f"echo {wandb_key}")
    wandb.login(key=wandb_api_key,relogin=True,force=True)

    # wandb.init(project='partial_spoof_demo')
    project_name='partial_spoof_AST_binary_classifier'

    sweep_config = {
        'method': 'bayes',
        'metric': 
        {
            'goal': 'minimize', 
            'name': 'validation_utterance_eer_epoch'
            },
        'parameters': 
        {
            # 'NUM_EPOCHS': {'values': [5, 7]},
            # 'LEARNING_RATE': {'values': [0.001]},
            # 'BATCH_SIZE': {'values': [16,32]},
            'NUM_EPOCHS': {'values': [1]},
            'LEARNING_RATE': {'values': [2e-4]},
            # 'LEARNING_RATE': {'values': [0.00021195579137608126]},
            # 'LEARNING_RATE': {'values': [2.3550643486231242e-05]},
            'BATCH_SIZE': {'values': [4]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config,project=project_name)
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=1)




if __name__ == "__main__":
    # Record the start time
    start_time = datetime.now()

    main()

    #========================= test inference ================
    # inference(PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/model_epochs60_batch8_lr0.005_20241226_214707.pth'))
    eval_audio_conf = {
    'num_mel_bins': 128,
    'freqm': 0,  # frequency masking parameter
    'timem': 0,  # time masking parameter
    'target_length': 1024,  # Target length for spectrogram
    }
    # inference(input_fdim=128,input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384',eval_audio_conf=eval_audio_conf,
    #     eval_data_path=os.path.join(os.getcwd(),'database/eval/con_wav'),
    #     eval_labels_path = os.path.join(os.getcwd(),'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json'),
    #     AST_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/AST_model_epochs30_batch4_lr0.0002_20250111_100116.pth'),
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





