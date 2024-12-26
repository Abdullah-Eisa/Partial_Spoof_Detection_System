whoami

apt update
apt install sudo


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

# when using root/container disk 
~/miniconda3/bin/conda init
# when using workspace/network disk 
# /workspace/miniconda3/bin/conda init

# restart/close and reopen the terminal
conda --version


# Get the current working directory
PWD=$(pwd)

# Set permissions recursively
chmod -R 777 "${PWD}"
# git restore --source HEAD~1 cloud_instance_pip_requirements.txt
# git restore --source HEAD~1 train_Wav2Vec2Conformer.py
# git restore --source HEAD~1 cloud_instance_requirements_2.txt
# git restore --source HEAD~1 environment.yml
# find "${PWD}" -type f  ! -name '*.txt' ! -name '*.yml' -exec chmod 777 {} \;

echo "Running env_setup.sh ..."
sh ./env_setup.sh

# install screen
sudo apt update
sudo apt install screen

rm Miniconda3-latest-Linux-x86_64.sh
echo "Done"
