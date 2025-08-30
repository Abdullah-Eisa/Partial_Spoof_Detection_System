
import os
import yaml
from pathlib import Path
import torch

class ConfigManager:
    def __init__(self, config_path=None):
        """Initialize configuration manager"""
        self.base_dir = Path(__file__).parent.parent.absolute()
        self.config_path = config_path or os.path.join(self.base_dir, 'config/default_config.yaml')
        self.config = self._load_config()
        self._process_paths()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Replace ${BASE_DIR} with actual path in all strings recursively
        def replace_base_dir(obj):
            if isinstance(obj, dict):
                return {k: replace_base_dir(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [replace_base_dir(elem) for elem in obj]
            elif isinstance(obj, str):
                return obj.replace('${BASE_DIR}', str(self.base_dir))
            return obj
        
        return replace_base_dir(config)

    def _process_paths(self):
        """Process and validate paths in configuration"""
        # Create required directories
        os.makedirs(self.config['paths']['model_save_dir'], exist_ok=True)
        
        # Override device based on availability
        if self.config['system']['device'] == 'cuda' and not torch.cuda.is_available():
            self.config['system']['device'] = 'cpu'

    def _validate_config(self):
        """Validate configuration values"""
        required_paths = [
            self.config['data']['train_data_path'],
            self.config['data']['train_labels_path'],
            self.config['data']['dev_data_path'],
            self.config['data']['dev_labels_path'],
            self.config['paths']['ssl_checkpoint'],
            self.config['paths']['ps_model_checkpoint']  # Add PS model checkpoint path
        ]
        
        missing_paths = []
        for path in required_paths:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print("Warning: The following required paths are missing:")
            for path in missing_paths:
                print(f"  - {path}")
            print("\nPlease ensure all required files and directories exist.")

    def get_wandb_key(self):
        """Get Weights & Biases API key if exists"""
        key_file = self.config['paths']['wandb_key_file']
        if os.path.exists(key_file):
            with open(key_file, 'r') as f:
                return f.read().strip()
        return None

    def __getitem__(self, key):
        """Allow dictionary-like access to config"""
        return self.config[key]


