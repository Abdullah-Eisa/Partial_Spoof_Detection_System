# Create and switch to a new branch (e.g., "new-branch-name")
# git checkout -b main
git checkout -b remove-max-pooling-block

# Initialize the Git repository
git init

# git rm --cached train_no_wandb.py                         # when removing a file locally and want to delete it on the remote repositry
# Add all files to the staging area
git add *

# Commit your changes
git commit -m " remove-max-pooling-block, update cloud_instance_setup.sh , inference.py , model.py , utils.py "
# Add the remote repository
git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Push the new branch to GitHub and set it to track the remote branch
git push -u origin remove-max-pooling-block
