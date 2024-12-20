git config --global user.email "s-abdallah@zewailcity.edu.eg"
git config --global user.name "Abdullah"

# Create and switch to a new branch (e.g., "new-branch-name")
git checkout -b cloud_instance_0


# Initialize the Git repository
git init

# Add all files to the staging area
git add *

# Commit your changes
git commit -m " modify torch versions in env_setup.sh , modify train_binary_classifier.py with the following
try combining these two lines in custom_collate_fn to one line waveforms = [item['waveform'] for item in batch] waveforms_padded=pad_sequence([waveform.squeeze(0) for waveform in waveforms], batch_first=True)
update/remove num_classes here PS_Model = BinarySpoofingClassificationModel(feature_dim=hidd_dims['wav2vec2'], num_heads=8, hidden_dim=128, num_classes=33,conformer_layers=1)
change audiodataset with 
waveform, sample_rate = torchaudio.load(file_path, normalize=False)
normalization of waveform = waveform / waveform.abs().max()
test on mini training e.g. 5 batches ,  Ensure torch.cuda.empty_cache() is called at strategic points (e.g., after each epoch) to free up unused memory, especially if you're using CUDA. python Copy code
test adding the following loss = criterion(outputs, labels) if torch.isnan(loss).any(): print(fNaN detected in loss at epoch {epoch}, batch {batch_idx}) continue
test training speed when updating mp.set_start_method('fork', force=True) vs when commenting it each on one epoch

"
# Add the remote repository
git remote add origin https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Push the new branch to GitHub and set it to track the remote branch
git push -u origin cloud_instance_0



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