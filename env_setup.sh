
#!/bin/bash

# ==================== ENV with GPU ==============================
# git clone -b <branch> <remote_repo>
# git clone -b cloud_instance_0  https://github.com/Abdullah-Eisa/Partial_Spoof_Detection_System.git
# conda create -n ${ENVNAME} python=3.9 pip --yes
# conda activate ${ENVNAME}
# conda env create -f environment.yml

# Install dependency for ps_scratch environment

# Name of the conda environment
ENVNAME=ps2
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
    # pip cache purge
    # pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
    # pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   # tested , do not work , pytorch libraries are not installed
    # pip install torch==2.2.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121   # tested , do not work , pytorch libraries are not installed
    
    # CUDA 11.8
    # conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
    # CUDA 12.1
    # conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia      # tested ,  works 
    conda install pytorch==2.2.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia         # slow & There appear to be 1 leaked semaphore objects to clean up at shutdown , may try pytorch==2.5.1 torchaudio==2.5.1
    # conda install pytorch==2.2.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia         # tested ,  works
    # conda install pytorch==2.5.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia         # tested , do not work , pytorch libraries are not installed
    # conda install pytorch==2.5.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia         # tested , do not work , pytorch libraries are not installed
    # Latest PyTorch requires Python 3.9 or later.
    # pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121            # tested , do not work , pytorch libraries are not installed


    # install some foundational libraries
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
        # from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForSequenceClassification
        # # Load the pre-trained model and processor
        # # model_name = "facebook/wav2vec2-conformer-rope-large-960h-ft"  # Example model name
        # model_name = "facebook/wav2vec2-conformer-rel-pos-large-960h-ft"  # Example model name
        # processor = Wav2Vec2Processor.from_pretrained(model_name)
        # model = Wav2Vec2ConformerForSequenceClassification.from_pretrained(model_name)

        # # Save them locally
        # processor.save_pretrained("./models/Wav2Vec2Processor")
        # model.save_pretrained("./models/Wav2Vec2ConformerForSequenceClassificationModel")
    #     EOF
    # )

    # python -c "$PYCMD"


    

else
    echo "Conda environment ${ENVNAME} has been installed"
fi