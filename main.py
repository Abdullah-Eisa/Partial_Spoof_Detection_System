
import wandb
import os
import random

from train import train

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
            'name': 'validation_utterance_eer_epoch'
            },
        'parameters': 
        {
            # 'NUM_EPOCHS': {'values': [5, 7]},
            # 'LEARNING_RATE': {'values': [0.001]},
            # 'BATCH_SIZE': {'values': [16,32]},
            'NUM_EPOCHS': {'values': [1]},
            'LEARNING_RATE': {'values': [0.001]},
            'BATCH_SIZE': {'values': [16]},
            # 'CLASS0_WEIGHT': {'values': [0.42,0.45,0.48]},

        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config,project='partial_spoof_trial_1')
    # sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id, function=train, count=2)




if __name__ == "__main__":
    main()






