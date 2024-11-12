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

echo "Running env_setup.sh ..."
sh ./env_setup.sh


echo "Done"
