# Create and switch to a new branch (e.g., "new-branch-name")
git checkout -b main

# Initialize the Git repository
git init

# git rm --cached train_no_wandb.py                         # when removing a file locally and want to delete it on the remote repositry
# Add all files to the staging area
git add *

# Commit your changes
git commit -m "   

truly modified files: 
        modified:   cloud_instance_setup.sh
        modified:   config/default_config.yaml
        modified:   preprocess.py
        modified:   utils/config_manager.py
        modified:   utils/utils.py


not truly:
        modified:   .gitignore
        modified:   .vscode/copilot-instructions.md
        modified:   cloud_instance_setup.sh
        modified:   config/default_config.yaml
        modified:   cross_dataset_evaluation.py
        modified:   environment.yml
        modified:   others/RFP_problems.md
        modified:   preprocess.py
        modified:   requirements.txt
        modified:   run_comprehensive_evaluation.py
        modified:   utils/__init__.py
        modified:   utils/attention_visualization.py
        modified:   utils/cluster_analysis.py
        modified:   utils/config_manager.py
        modified:   utils/generate_efficiency_table.py
        modified:   utils/gradient_analysis.py
        modified:   utils/parameter_counter.py
        modified:   utils/spoofing_algorithm_util.py
        modified:   utils/utils.py

 "
# Add the remote repository
git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Push the new branch to GitHub and set it to track the remote branch
git push -u origin main
