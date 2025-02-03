git config --global user.email "s-abdallah@zewailcity.edu.eg"
git config --global user.name "Abdullah"

# Create and switch to a new branch (e.g., "new-branch-name")
git checkout -b main


# Initialize the Git repository
git init

# git rm --cached train_no_wandb.py                         # when removing a file locally and want to delete it on the remote repositry
# Add all files to the staging area
git add *

# Commit your changes
git commit -m "modify the last commit from RFP_train branch to include dataloaders for different datasets , move model testing to inference.py , modify function parameters in the train.py and inference.py to be in json like format"
# Add the remote repository
git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Push the new branch to GitHub and set it to track the remote branch
git push -u origin main



# # echo "# Partial_Spoof_Detection_System" >> README.md
# git init
# git add *
# git commit -m "add model parallelization using torch.nn.DataParallel , correct save_interval "
# git branch -M main
# git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
# git push -u  origin main
# # git push -u --force origin main



# git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
# git branch -M main
# git push -u origin main


# echo "# Partial_Spoof_Detection_System" >> README.md
# git init
# git add README.md
# git commit -m ""
# git branch -M main
# git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
# git push -u origin main