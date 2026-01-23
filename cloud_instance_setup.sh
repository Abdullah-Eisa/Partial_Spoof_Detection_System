whoami

apt update
apt install sudo

# install screen
sudo apt update
sudo apt install screen

sudo apt update
sudo apt install unzip


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

# when using root/container disk 
~/miniconda3/bin/conda init
# when using workspace/network disk 
# /workspace/miniconda3/bin/conda init

# restart/close and reopen the terminal
conda --version

# conda init bash && source ~/.bashrc 

# Get the current working directory
PWD=$(pwd)

# Set permissions recursively
chmod -R 777 "${PWD}"
# git restore --source HEAD~1 \
#   .gitignore \
#   .vscode/copilot-instructions.md \
#   environment.yml \
#   others/RFP_problems.md \
#   requirements.txt \
#   run_comprehensive_evaluation.py \
#   cross_dataset_evaluation.py \
#   utils/__init__.py \
#   utils/attention_visualization.py \
#   utils/cluster_analysis.py \
#   utils/config_manager.py \
#   utils/generate_efficiency_table.py \
#   utils/gradient_analysis.py \
#   utils/parameter_counter.py \
#   utils/spoofing_algorithm_util.py



# find "${PWD}" -type f  ! -name '*.txt' ! -name '*.yml' -exec chmod 777 {} \;

echo "Running env_setup.sh ..."
bash ./env_setup.sh


rm Miniconda3-latest-Linux-x86_64.sh
echo "Done"