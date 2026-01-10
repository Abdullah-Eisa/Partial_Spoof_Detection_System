#!/usr/bin/env python3
"""
Multi-Group Training Script for Partial Spoof Detection System

This script trains the model sequentially on three data groups:
  - 2F4R: Two fake segments followed by four real segments
  - 4R2F: Four real segments followed by two fake segments
  - 2R2F2R: Two real, two fake, two real segments

Usage:
    python run_all_groups.py                    # Train all groups
    python run_all_groups.py --groups 2F4R      # Train specific group only
    python run_all_groups.py --groups 2F4R 4R2F # Train specific groups

Author: Generated for Partial Spoof Detection System
Date: January 2026
"""

import os
import sys
import yaml
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import time


class MultiGroupTrainer:
    """
    Manages sequential training across multiple data groups.
    Handles config updates, training execution, and result tracking.
    """
    
    def __init__(self, base_dir=None):
        """
        Initialize the multi-group trainer.
        
        Args:
            base_dir: Base directory of the project (default: current directory)
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.config_file = self.base_dir / 'config' / 'default_config.yaml'
        self.backup_file = self.base_dir / 'config' / 'default_config_backup.yaml'
        
        # Define data groups with their configurations
        self.groups = {
            '2F4R': {
                'name': '2F4R',
                'description': 'Two fake segments followed by four real segments',
                # 'dataset_name': 'PF_Detection_by_Segment_Location_Dataset_2F4R',
                'dataset_name': 'PF_Detection_by_Segment_Location_Dataset',
                'paths': {
                    'train_data': 'database/Rfp_Test/2F4R/training',
                    'train_labels': 'database/Rfp_Test/2F4R/labels/2F4R__training_subset_labels.txt',
                    'dev_data': 'database/Rfp_Test/2F4R/validation',
                    'dev_labels': 'database/Rfp_Test/2F4R/labels/2F4R__validation_subset_labels.txt',
                    'eval_data': 'database/Rfp_Test/2F4R/testing',
                    'eval_labels': 'database/Rfp_Test/2F4R/labels/2F4R__testing_subset_labels.txt',
                }
            },
            '4R2F': {
                'name': '4R2F',
                'description': 'Four real segments followed by two fake segments',
                # 'dataset_name': 'PF_Detection_by_Segment_Location_Dataset_4R2F',
                'dataset_name': 'PF_Detection_by_Segment_Location_Dataset',
                'paths': {
                    'train_data': 'database/Rfp_Test/4R2F/training',
                    'train_labels': 'database/Rfp_Test/4R2F/labels/4R2F__training_subset_labels.txt',
                    'dev_data': 'database/Rfp_Test/4R2F/validation',
                    'dev_labels': 'database/Rfp_Test/4R2F/labels/4R2F__validation_subset_labels.txt',
                    'eval_data': 'database/Rfp_Test/4R2F/testing',
                    'eval_labels': 'database/Rfp_Test/4R2F/labels/4R2F__testing_subset_labels.txt',
                }
            },
            '2R2F2R': {
                'name': '2R2F2R',
                'description': 'Two real, two fake, two real segments',
                # 'dataset_name': 'PF_Detection_by_Segment_Location_Dataset_2R2F2R',
                'dataset_name': 'PF_Detection_by_Segment_Location_Dataset',
                'paths': {
                    'train_data': 'database/Rfp_Test/2R2F2R/training',
                    'train_labels': 'database/Rfp_Test/2R2F2R/labels/2R2F2R__training_subset_labels.txt',
                    'dev_data': 'database/Rfp_Test/2R2F2R/validation',
                    'dev_labels': 'database/Rfp_Test/2R2F2R/labels/2R2F2R__validation_subset_labels.txt',
                    'eval_data': 'database/Rfp_Test/2R2F2R/testing',
                    'eval_labels': 'database/Rfp_Test/2R2F2R/labels/2R2F2R__testing_subset_labels.txt',
                }
            }
        }
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.base_dir / 'outputs' / f'multi_group_training_{timestamp}'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results tracking
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        print(f"\n{'='*80}")
        print(f"Multi-Group Training Initialized")
        print(f"{'='*80}")
        print(f"Base Directory: {self.base_dir}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Config File: {self.config_file}")
        print(f"{'='*80}\n")
    
    def verify_prerequisites(self):
        """Verify all required files and directories exist."""
        print("🔍 Verifying prerequisites...")
        
        issues = []
        
        # Check config file
        if not self.config_file.exists():
            issues.append(f"Config file not found: {self.config_file}")
        
        # Check data directories and labels for each group
        for group_name, group_info in self.groups.items():
            paths = group_info['paths']
            
            # Check training data
            train_data_path = self.base_dir / paths['train_data']
            if not train_data_path.exists():
                issues.append(f"[{group_name}] Training data not found: {train_data_path}")
            
            # Check training labels
            train_labels_path = self.base_dir / paths['train_labels']
            if not train_labels_path.exists():
                issues.append(f"[{group_name}] Training labels not found: {train_labels_path}")
            
            # Check validation data
            dev_data_path = self.base_dir / paths['dev_data']
            if not dev_data_path.exists():
                issues.append(f"[{group_name}] Validation data not found: {dev_data_path}")
            
            # Check validation labels
            dev_labels_path = self.base_dir / paths['dev_labels']
            if not dev_labels_path.exists():
                issues.append(f"[{group_name}] Validation labels not found: {dev_labels_path}")
        
        if issues:
            print("\n❌ Prerequisites check failed:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        print("✅ All prerequisites verified!\n")
        return True
    
    def backup_config(self):
        """Create a backup of the original configuration file."""
        print(f"💾 Creating config backup...")
        print(f"   Source: {self.config_file}")
        print(f"   Backup: {self.backup_file}")
        
        try:
            shutil.copy2(self.config_file, self.backup_file)
            print("   ✅ Backup created successfully\n")
            return True
        except Exception as e:
            print(f"   ❌ Backup failed: {str(e)}\n")
            return False
    
    def restore_config(self):
        """Restore the original configuration from backup."""
        print(f"\n📂 Restoring original config...")
        print(f"   Source: {self.backup_file}")
        print(f"   Target: {self.config_file}")
        
        if self.backup_file.exists():
            try:
                shutil.copy2(self.backup_file, self.config_file)
                print("   ✅ Config restored successfully")
                
                # Remove backup file
                self.backup_file.unlink()
                print("   ✅ Backup file removed\n")
                return True
            except Exception as e:
                print(f"   ❌ Restore failed: {str(e)}\n")
                return False
        else:
            print(f"   ⚠️  Backup file not found: {self.backup_file}\n")
            return False
    
    def update_config(self, group_name):
        """
        Update config file with group-specific paths.
        
        Args:
            group_name: Name of the group (e.g., '2F4R')
        """
        print(f"🔧 Updating config for group: {group_name}")
        
        try:
            group = self.groups[group_name]
            
            # Load current config
            with open(self.config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Update dataset name
            config['data']['dataset_name'] = group['dataset_name']
            
            # Update data paths
            paths = group['paths']
            config['data']['train_data_path'] = f"${{BASE_DIR}}/{paths['train_data']}"
            config['data']['train_labels_path'] = f"${{BASE_DIR}}/{paths['train_labels']}"
            config['data']['dev_data_path'] = f"${{BASE_DIR}}/{paths['dev_data']}"
            config['data']['dev_labels_path'] = f"${{BASE_DIR}}/{paths['dev_labels']}"
            config['data']['eval_data_path'] = f"${{BASE_DIR}}/{paths['eval_data']}"
            config['data']['eval_labels_path'] = f"${{BASE_DIR}}/{paths['eval_labels']}"
            
            # Update model save directory to be group-specific
            config['paths']['model_save_dir'] = f"${{BASE_DIR}}/models/back_end_models/{group_name}"
            
            # Save updated config
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"   ✅ Config updated successfully")
            print(f"      Dataset: {group['dataset_name']}")
            print(f"      Train: {paths['train_data']}")
            print(f"      Models: models/back_end_models/{group_name}\n")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Config update failed: {str(e)}\n")
            return False
    
    def run_training(self, group_name):
        """
        Execute training for a specific group.
        
        Args:
            group_name: Name of the group to train
            
        Returns:
            bool: True if training succeeded, False otherwise
        """
        group = self.groups[group_name]
        
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING GROUP: {group_name}")
        print(f"{'='*80}")
        print(f"Description: {group['description']}")
        print(f"Dataset: {group['dataset_name']}")
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")
        
        # Update config for this group
        if not self.update_config(group_name):
            return False
        
        # Create log file
        log_file = self.output_dir / f'group_{group_name}.log'
        print(f"📝 Logging to: {log_file}\n")
        
        # Track group start time
        group_start_time = datetime.now()
        
        # Run training
        try:
            print(f"▶️  Executing: python main.py\n")
            print(f"{'─'*80}\n")
            
            with open(log_file, 'w', buffering=1) as log:
                # Write header to log
                log.write(f"{'='*80}\n")
                log.write(f"Training Log: {group_name}\n")
                log.write(f"Start Time: {group_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"Description: {group['description']}\n")
                log.write(f"{'='*80}\n\n")
                log.flush()
                
                # Execute main.py
                process = subprocess.Popen(
                    [sys.executable, 'main.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1,
                    cwd=self.base_dir
                )
                
                # Stream output in real-time
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                
                # Wait for process to complete
                return_code = process.wait()
                
                # Calculate duration
                group_end_time = datetime.now()
                duration = group_end_time - group_start_time
                
                # Write footer to log
                log.write(f"\n{'='*80}\n")
                log.write(f"End Time: {group_end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log.write(f"Duration: {duration}\n")
                log.write(f"Return Code: {return_code}\n")
                log.write(f"{'='*80}\n")
                
                # Check success
                success = (return_code == 0)
                
                print(f"\n{'─'*80}\n")
                
                if success:
                    print(f"✅ Group {group_name} completed successfully")
                    print(f"   Duration: {duration}")
                else:
                    print(f"❌ Group {group_name} failed (return code: {return_code})")
                    print(f"   Duration: {duration}")
                
                print(f"   Log file: {log_file}")
                print(f"{'='*80}\n")
                
                return success
                
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Training interrupted by user (Ctrl+C)")
            print(f"   Group {group_name} was not completed")
            return False
            
        except Exception as e:
            print(f"\n❌ Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_metrics(self, log_file):
        """
        Extract performance metrics from training log.
        
        Args:
            log_file: Path to log file
            
        Returns:
            dict: Extracted metrics or None
        """
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            metrics = {}
            
            # Look for utterance EER
            import re
            
            # Pattern: "Average ... Utterance EER: 0.1234"
            eer_pattern = r'Utterance EER:\s*([0-9.]+)'
            matches = re.findall(eer_pattern, content)
            if matches:
                metrics['utterance_eer'] = float(matches[-1])  # Get last occurrence
            
            # Pattern: "Utterance EER Threshold: 0.5678"
            threshold_pattern = r'Utterance EER Threshold:\s*([0-9.]+)'
            matches = re.findall(threshold_pattern, content)
            if matches:
                metrics['eer_threshold'] = float(matches[-1])
            
            # Pattern: "Test Loss: 0.1234"
            loss_pattern = r'Test Loss:\s*([0-9.]+)'
            matches = re.findall(loss_pattern, content)
            if matches:
                metrics['test_loss'] = float(matches[-1])
            
            return metrics if metrics else None
            
        except Exception as e:
            print(f"   ⚠️  Could not extract metrics: {str(e)}")
            return None
    
    def run_all_groups(self, groups_to_run=None, wait_between=30):
        """
        Run training on all specified groups sequentially.
        
        Args:
            groups_to_run: List of group names (default: all groups)
            wait_between: Seconds to wait between groups (default: 30)
        """
        # Use all groups if none specified
        if groups_to_run is None:
            groups_to_run = list(self.groups.keys())
        
        # Validate group names
        invalid_groups = [g for g in groups_to_run if g not in self.groups]
        if invalid_groups:
            print(f"❌ Invalid group names: {invalid_groups}")
            print(f"   Valid groups: {list(self.groups.keys())}")
            return
        
        # Print header
        print(f"\n{'#'*80}")
        print(f"# MULTI-GROUP TRAINING")
        print(f"{'#'*80}")
        print(f"Groups to train: {', '.join(groups_to_run)}")
        print(f"Total groups: {len(groups_to_run)}")
        print(f"Wait between groups: {wait_between} seconds")
        print(f"{'#'*80}\n")
        
        # Verify prerequisites
        if not self.verify_prerequisites():
            print("❌ Prerequisites check failed. Aborting.")
            return
        
        # Backup config
        if not self.backup_config():
            print("❌ Config backup failed. Aborting.")
            return
        
        # Track overall start time
        self.start_time = datetime.now()
        
        try:
            # Train each group
            for i, group_name in enumerate(groups_to_run, 1):
                print(f"\n{'#'*80}")
                print(f"# PROGRESS: {i}/{len(groups_to_run)}")
                print(f"# GROUP: {group_name}")
                print(f"{'#'*80}")
                
                # Run training
                success = self.run_training(group_name)
                self.results[group_name] = success
                
                # Handle failure
                if not success:
                    print(f"\n⚠️  Group {group_name} failed!")
                    
                    # Ask user if they want to continue
                    try:
                        response = input("\nContinue with remaining groups? (y/n): ").strip().lower()
                        if response != 'y':
                            print("Stopping training.")
                            break
                    except KeyboardInterrupt:
                        print("\n\nTraining interrupted by user.")
                        break
                
                # Wait between groups (except after last group)
                if i < len(groups_to_run):
                    print(f"\n⏸️  Waiting {wait_between} seconds before next group...")
                    try:
                        time.sleep(wait_between)
                    except KeyboardInterrupt:
                        print("\n\nWait interrupted. Starting next group immediately.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        
        finally:
            # Track overall end time
            self.end_time = datetime.now()
            
            # Always restore config
            self.restore_config()
            
            # Print summary
            self.print_summary()
    
    def print_summary(self):
        """Print comprehensive training summary."""
        print(f"\n{'='*80}")
        print(f"📊 TRAINING SUMMARY")
        print(f"{'='*80}")
        
        if not self.results:
            print("No training results available.")
            print(f"{'='*80}\n")
            return
        
        # Calculate statistics
        total_groups = len(self.results)
        successful_groups = sum(self.results.values())
        failed_groups = total_groups - successful_groups
        
        # Print results for each group
        print(f"\nResults by Group:")
        print(f"{'-'*80}")
        print(f"{'Group':<15} {'Status':<15} {'EER':<12} {'Threshold':<12} {'Loss':<12}")
        print(f"{'-'*80}")
        
        for group_name, success in self.results.items():
            status = "✅ Success" if success else "❌ Failed"
            
            # Try to extract metrics from log
            log_file = self.output_dir / f'group_{group_name}.log'
            metrics = self.extract_metrics(log_file) if log_file.exists() else None
            
            eer = f"{metrics['utterance_eer']:.4f}" if metrics and 'utterance_eer' in metrics else "N/A"
            threshold = f"{metrics['eer_threshold']:.4f}" if metrics and 'eer_threshold' in metrics else "N/A"
            loss = f"{metrics['test_loss']:.4f}" if metrics and 'test_loss' in metrics else "N/A"
            
            print(f"{group_name:<15} {status:<15} {eer:<12} {threshold:<12} {loss:<12}")
        
        # Print overall statistics
        print(f"{'-'*80}")
        print(f"\nOverall Statistics:")
        print(f"  Total Groups:      {total_groups}")
        print(f"  ✅ Successful:      {successful_groups}")
        print(f"  ❌ Failed:          {failed_groups}")
        
        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\nTiming:")
            print(f"  Start Time:        {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End Time:          {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Total Duration:    {total_duration}")
        
        print(f"\n📁 Output Location:")
        print(f"  Logs:              {self.output_dir}")
        print(f"  Models:            models/back_end_models/{{group_name}}/")
        
        print(f"\n{'='*80}")
        
        # Final message
        if successful_groups == total_groups:
            print(f"🎉 All training completed successfully!")
        elif successful_groups > 0:
            print(f"⚠️  Training completed with some failures.")
        else:
            print(f"❌ All training attempts failed.")
        
        print(f"{'='*80}\n")


def main():
    """Main entry point for the script."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Sequential training on multiple data groups',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_groups.py                    # Train all groups
  python run_all_groups.py --groups 2F4R      # Train only 2F4R
  python run_all_groups.py --groups 2F4R 4R2F # Train specific groups
  python run_all_groups.py --wait 60          # Wait 60s between groups
        """
    )
    
    parser.add_argument(
        '--groups',
        nargs='+',
        choices=['2F4R', '4R2F', '2R2F2R'],
        help='Specific groups to train (default: all groups)'
    )
    
    parser.add_argument(
        '--wait',
        type=int,
        default=30,
        help='Seconds to wait between groups (default: 30)'
    )
    
    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory of the project (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Create trainer instance
    try:
        trainer = MultiGroupTrainer(base_dir=args.base_dir)
    except Exception as e:
        print(f"❌ Failed to initialize trainer: {str(e)}")
        sys.exit(1)
    
    # Run training
    try:
        trainer.run_all_groups(
            groups_to_run=args.groups,
            wait_between=args.wait
        )
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Exit with appropriate code
    if trainer.results:
        all_success = all(trainer.results.values())
        sys.exit(0 if all_success else 1)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()