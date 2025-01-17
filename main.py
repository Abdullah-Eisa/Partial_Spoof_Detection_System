
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
    project_name='RFP_Wav2Vec2_Conformer_binary_classifier'

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
            'NUM_EPOCHS': {'values': [30]},
            'LEARNING_RATE': {'values': [0.0001]},
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

    # inference(eval_data_path=os.path.join(os.getcwd(),'database/RFP/testing'),
    #     eval_labels_path = os.path.join(os.getcwd(),'database/RFP/labels/ASVspoof2017_V2_eval.trl.txt'),
    #     ssl_ckpt_path=os.path.join(os.getcwd(), 'models/w2v_large_lv_fsh_swbd_cv.pt'),
    #     PS_Model_path=os.path.join(os.getcwd(),f'models/back_end_models/???????.pth'),
    #     feature_dim=768, num_heads=8, hidden_dim=128, max_dropout=0, depthwise_conv_kernel_size=31,
    #     conformer_layers=1, max_pooling_factor=3,
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





