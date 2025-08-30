
import os
from datetime import datetime
import wandb
from utils.config_manager import ConfigManager
from train import train

def main():
    """Main execution function"""
    # Load configuration
    config = ConfigManager()
    print(f"✨✨✨✨✨ config['training']['use_wandb']= {config['training']['use_wandb']}")

    if config['training']['use_wandb']:
        # Initialize wandb with sweep config
        wandb_key = config.get_wandb_key()
        if wandb_key:
            wandb.login(key=wandb_key, relogin=True, force=True)
            
            sweep_config = {
                'method': 'bayes',
                'metric': {
                    'goal': 'minimize',
                    'name': 'validation_utterance_eer_epoch'
                },
                'parameters': config['wandb_sweep']  # Move sweep parameters to config
            }
            
            project_name = f"{config['data']['dataset_name']}_Wav2Vec2_Conformer_binary_classifier"
            sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
            wandb.agent(sweep_id, function=lambda: train(config), count=1)
        else:
            print("Warning: WandB enabled but API key not found. Running without WandB.")
            train(config)
    else:
        train(config)

if __name__ == "__main__":
    start_time = datetime.now()
    main()
    end_time = datetime.now()
    
    total_time = end_time - start_time
    hours, remainder = divmod(total_time.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"Total time: {int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds")


