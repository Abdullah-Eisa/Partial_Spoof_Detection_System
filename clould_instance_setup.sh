whoami

apt update
apt install sudo


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

~/miniconda3/bin/conda init

conda --version


chmod -R 777 /path/to/your/folder



from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

# Define the model name
model_name = "facebook/wav2vec2-base-960h"

# Load the tokenizer and model
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForCTC.from_pretrained(model_name)

# Save them locally
tokenizer.save_pretrained("./models/local_wav2vec2_tokenizer")
model.save_pretrained("./models/local_wav2vec2_model")