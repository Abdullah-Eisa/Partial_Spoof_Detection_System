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



    PYCMD=$(cat <<EOF
    from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

    # Define the model name
    model_name = "facebook/wav2vec2-base-960h"

    # Load the tokenizer and model
    tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Save them locally
    tokenizer.save_pretrained("./models/local_wav2vec2_tokenizer")
    model.save_pretrained("./models/local_wav2vec2_model")
    EOF
    )

    python -c "$PYCMD"


    

else
    echo "Conda environment ${ENVNAME} has been installed"
fi







# # ==================== ENV with CPU ==============================

# # Install dependency for ps_scratch environment

# # Name of the conda environment
# ENVNAME=ps_scratch
# REQUIREMENT_FILE=ps_scratch_requirements.txt

# eval "$(conda shell.bash hook)"

# # conda create --name ${ENVNAME} --file ${REQUIREMENT_FILE}
# conda activate ${ENVNAME}
# retVal=$?
# if [ $retVal -ne 0 ]; then
#     echo "Install conda environment ${ENVNAME}"
    
#     # conda env
#     conda create -n ${ENVNAME} python=3.9 pip --yes --file ${REQUIREMENT_FILE}
#     conda activate ${ENVNAME}



#     # make empty folders if not available
#     python -c "import os; os.makedirs("database", exist_ok=True)"
#     python -c "import os; os.makedirs("models", exist_ok=True)"
#     python -c "import os; os.makedirs("outputs", exist_ok=True)"



# else
#     echo "Conda environment ${ENVNAME} has been installed"
# fi



# ==================================================

# #!/bin/bash
# # Install dependency for fairseq

# # Name of the conda environment
# ENVNAME=ps_scratch

# eval "$(conda shell.bash hook)"
# conda activate ${ENVNAME}
# retVal=$?
# if [ $retVal -ne 0 ]; then
#     echo "Install conda environment ${ENVNAME}"
    
#     # conda env
#     conda create -n ${ENVNAME} python=3.9 pip --yes
#     conda activate ${ENVNAME}

#     # install pytorch
#     echo "===========Install pytorch==========="
#     # conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
#     # pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#     # pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
#     pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu


#     # install transformers
#     pip install transformers
    
#     # git clone fairseq
#     #  fairseq 0.10.2 on pip does not work
#     # git clone https://github.com/pytorch/fairseq
#     # cd fairseq
#     # pip install git+https://github.com/facebookresearch/fairseq.git@a54021305d6b3c4c5959ac9395135f63202db8f1

#     # install scipy
#     # pip install scipy==1.7.3

#     # install pandas
#     # pip install pandas==1.3.5

#     # install protobuf
#     # pip install protobuf==3.20.3

#     # install tensorboard
#     # pip install tensorboard==2.6.0
#     # pip install tensorboardX==2.6

#     # install librosa
#     # pip install librosa==0.10.0
#     pip install librosa

#     # install pydub
#     # pip install pydub==0.25.1

# else
#     echo "Conda environment ${ENVNAME} has been installed"
# fi

