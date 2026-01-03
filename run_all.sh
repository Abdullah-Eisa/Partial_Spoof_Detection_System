# # Create and switch to a new branch (e.g., "new-branch-name")
# git checkout -b main

# # Initialize the Git repository
# git init

# # git rm --cached train_no_wandb.py                         # when removing a file locally and want to delete it on the remote repositry
# # Add all files to the staging area
# git add *

# # Commit your changes
# git commit -m " update preprocess.py to handle fallback with zero value when loading dataset , update README.md to include run training with logs cmd "
# # Add the remote repository
# git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# # Push the new branch to GitHub and set it to track the remote branch
# git push -u origin main



# Create and switch to a NEW branch
# git checkout -b Feature-Extractor-Comparison
git checkout -b Sequence-Modeling-Alternatives

# Stage changes
git add .

# Commit changes
git commit -m "add working sequence models alternative Options: 'conformer', 'lstm', 'transformer', 'cnn' , dummy test on the ASVSpoof19 database "

# Push new branch to remote and set upstream
# git push -u origin Feature-Extractor-Comparison
git push -u origin Sequence-Modeling-Alternatives
