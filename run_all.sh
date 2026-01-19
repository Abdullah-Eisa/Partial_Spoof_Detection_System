# Create and switch to a new branch (e.g., "new-branch-name")
git checkout -b main

# Initialize the Git repository
git init

# git rm --cached train_no_wandb.py                         # when removing a file locally and want to delete it on the remote repositry
# Add all files to the staging area
git add *

# Commit your changes
git commit -m "   
add 
        utils/parameter_counter.py
        utils/spoofing_algorithm_util.py

        modified:   config/default_config.yaml
        deleted:    utils/reporting_utils copy 2.py
        deleted:    utils/reporting_utils copy 3.py
        deleted:    utils/reporting_utils copy 4.py
        deleted:    utils/reporting_utils copy 5.py
        deleted:    utils/reporting_utils copy 6.py
        deleted:    utils/reporting_utils copy 7.py
        deleted:    utils/reporting_utils copy.py

 "
# Add the remote repository
git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Push the new branch to GitHub and set it to track the remote branch
git push -u origin main
