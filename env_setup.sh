#!/bin/bash

# ==================== ENV with GPU ==============================
# git clone -b <branch> <remote_repo>
# git clone -b cloud_instance_0  https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git

# Install dependency for ps_scratch environment

# Name of the conda environment
ENVNAME=ps_scratch
# REQUIREMENT_FILE=ps_scratch_requirements.txt

eval "$(conda shell.bash hook)"

# conda create --name ${ENVNAME} --file ${REQUIREMENT_FILE}
conda activate ${ENVNAME}
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"
    
    # conda env
    conda create -n ${ENVNAME} python=3.9 pip --yes
    conda activate ${ENVNAME}

    # install pytorch
    echo "===========Install pytorch==========="
    # pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121


    # install some foundational libraries
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
    python -c "import os; os.makedirs('models', exist_ok=True)"
    python -c "import os; os.makedirs('models/back_end_models', exist_ok=True)"
    python -c "import os; os.makedirs('outputs', exist_ok=True)"



    # PYCMD=$(cat <<EOF
    #     from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

    #     # Define the model name
    #     model_name = "facebook/wav2vec2-base-960h"

    #     # Load the tokenizer and model
    #     tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    #     model = Wav2Vec2ForCTC.from_pretrained(model_name)

    #     # Save them locally
    #     tokenizer.save_pretrained("./models/local_wav2vec2_tokenizer")
    #     model.save_pretrained("./models/local_wav2vec2_model")
    #     EOF
    # )

    # python -c "$PYCMD"


    

else
    echo "Conda environment ${ENVNAME} has been installed"
fi

