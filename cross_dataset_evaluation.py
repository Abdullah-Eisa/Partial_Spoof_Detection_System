
"""
Cross-Dataset Evaluation Script
Evaluates trained model on all three datasets separately
"""

import torch
from utils.config_manager import ConfigManager
from inference import inference_helper
from model import initialize_models
from preprocess import initialize_data_loader
import json
from datetime import datetime

def cross_dataset_evaluation(config):
    """
    Evaluate model on all three datasets separately.
    
    Returns:
        Dictionary with results for each dataset
    """
    device = torch.device(config['system']['device'])
    
    # Load model
    print("Loading model...")
    model, feature_extractor, _ = initialize_models(
        ssl_ckpt_path=config['paths']['ssl_checkpoint'],
        save_feature_extractor=False,
        feature_dim=config['model']['feature_dim'],
        num_heads=config['model']['num_heads'],
        hidden_dim=config['model']['hidden_dim'],
        max_dropout=config['model']['max_dropout'],
        depthwise_conv_kernel_size=config['model']['depthwise_conv_kernel_size'],
        conformer_layers=config['model']['conformer_layers'],
        max_pooling_factor=config['model']['max_pooling_factor'],
        LEARNING_RATE=0.0001,
        DEVICE=device
    )
    
    checkpoint = torch.load(config['paths']['ps_model_checkpoint'], map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    feature_extractor.eval()
    
    # Initialize loss function
    from model import initialize_loss_function
    criterion = initialize_loss_function().to(device)
    
    # Define datasets to evaluate
    datasets = [
        {
            'name': 'RFP',
            'dataset_name': 'RFP_Dataset',
            'data_path': config['data'].get('rfp_eval_data_path', 
                                           'database/RFP/testing'),
            'labels_path': config['data'].get('rfp_eval_labels_path',
                                             'database/RFP/labels/ASVspoof2017_V2_eval.trl.txt')
        },
        {
            'name': 'PartialSpoof',
            'dataset_name': 'PartialSpoof_Dataset',
            'data_path': config['data'].get('ps_eval_data_path',
                                           'database/PartialSpoof/eval/con_wav'),
            'labels_path': config['data'].get('ps_eval_labels_path',
                                             'database/utterance_labels/PartialSpoof_LA_cm_eval_trl.json')
        },
        {
            'name': 'ASVspoof2019',
            'dataset_name': 'ASVspoof2019_LA_Dataset',
            'data_path': config['data'].get('asvspoof_eval_data_path',
                                           'database/ASVspoof2019/LA/ASVspoof2019_LA_eval/flac'),
            'labels_path': config['data'].get('asvspoof_eval_labels_path',
                                             'database/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt')
        }
    ]
    
    results = {}
    
    for dataset_info in datasets:
        print(f"\n{'='*80}")
        print(f"Evaluating on {dataset_info['name']}")
        print(f"{'='*80}")
        
        # Create data loader
        eval_loader = initialize_data_loader(
            dataset_name=dataset_info['dataset_name'],
            data_path=dataset_info['data_path'],
            labels_path=dataset_info['labels_path'],
            BATCH_SIZE=config['inference']['batch_size'],
            shuffle=False,
            num_workers=config['inference'].get('num_workers', 4),
            prefetch_factor=config['inference'].get('prefetch_factor', 2),
            pin_memory=config['inference'].get('pin_memory', True)
        )
        
        # Run inference
        metrics = inference_helper(
            model=model,
            feature_extractor=feature_extractor,
            criterion=criterion,
            test_data_loader=eval_loader,
            DEVICE=device
        )
        

        # Convert numpy / torch scalars to native Python types
        for k, v in metrics.items():
            if isinstance(v, (torch.Tensor,)):
                metrics[k] = v.item()
            elif hasattr(v, "item"):  # numpy scalar (e.g., float32)
                metrics[k] = v.item()


        results[dataset_info['name']] = metrics
        
        print(f"\nResults on {dataset_info['name']}:")
        print(f"  EER: {metrics['utterance_eer']:.4f} ({metrics['utterance_eer']*100:.2f}%)")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f'outputs/cross_dataset_results_{timestamp}.json'
    
    import os
    os.makedirs('outputs', exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    config = ConfigManager()
    results = cross_dataset_evaluation(config)
    
    print("\n" + "="*80)
    print("CROSS-DATASET EVALUATION SUMMARY")
    print("="*80)
    
    for dataset_name, metrics in results.items():
        print(f"\n{dataset_name}:")
        print(f"  EER: {metrics['utterance_eer']*100:.2f}%")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
        print(f"  F1: {metrics.get('f1', 0):.4f}")

