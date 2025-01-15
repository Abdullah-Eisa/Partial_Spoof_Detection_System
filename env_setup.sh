#!/bin/bash
# ==================== ENV with GPU ==============================
# git clone -b <branch> <remote_repo>
# git clone -b ASVspoof_train  https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
# ========================
# git fetch origin
# git reset --hard origin/<branch_name>
# conda create -n ${ENVNAME} python=3.9 pip --yes
# conda activate ${ENVNAME}
# conda env create -f environment.yml

# Install dependency for ps_scratch environment

# Name of the conda environment
ENVNAME=ps

eval "$(conda shell.bash hook)"

# conda create --name ${ENVNAME} --file ${REQUIREMENT_FILE}
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Installing conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.9 pip --yes
    conda activate ${ENVNAME}

    # install pytorch with compatible cuda and torchaudio
    echo "=========== Installing PyTorch and Torchaudio ==========="
    conda update conda
    pip install --upgrade pip
    conda clean --all


    pip install torch
    pip install torchaudio
    pip install s3prl
    pip install os
    pip install librosa
    pip install matplotlib
    pip install numpy
    pip install pathlib
    pip install protobuf
    pip install scikit-learn
    pip install scipy
    pip install setuptools
    pip install tqdm
    pip install transformers
    pip install wandb
    
    # make empty folders if not available
    python -c "import os; os.makedirs('database', exist_ok=True)"
    python -c "import os; os.makedirs('database/ASVspoof2019', exist_ok=True)"
    python -c "import os; os.makedirs('models', exist_ok=True)"
    python -c "import os; os.makedirs('models/back_end_models', exist_ok=True)"
    python -c "import os; os.makedirs('outputs', exist_ok=True)"
    
else
    echo "Conda environment ${ENVNAME} is already installed"
fi
