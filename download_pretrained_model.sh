# Description: Download the pre-trained wav2vec 2.0 model from the official fairseq repository
# code adapted from: https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/03multireso/01_download_pretrained_models.sh
# set -x

# ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
# save_folder="./models"  # Specify the folder here
# if [ ! -e ${save_folder}/${ssl_model} ]; then
#     wget --show-progress https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model} -P ${save_folder}
# fi

# hubert_model="hubert_large_ll60k.pt"
# if [ ! -e ${save_folder}/${hubert_model} ]; then
#     wget --show-progress https://dl.fbaipublicfiles.com/huggingface/pytorch-models/hubert/${hubert_model} -P ${save_folder}
# fi


# #!/bin/bash
# set -x

# save_folder="./models"
# mkdir -p "${save_folder}"

# echo "Select the model you want to download:"
# echo "1) wav2vec 2.0 (w2v_large_lv_fsh_swbd_cv)"
# echo "2) HuBERT (hubert_large_ll60k)"
# read -p "Enter your choice (1 or 2): " model_choice

# case $model_choice in
#     1)
#         ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
#         url="https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model}"

#         if [ ! -f "${save_folder}/${ssl_model}" ]; then
#             echo "Downloading wav2vec 2.0 model..."
#             curl -L "${url}" -o "${save_folder}/${ssl_model}"
#         else
#             echo "wav2vec 2.0 model already exists. Skipping download."
#         fi
#         ;;
#     2)
#         hubert_model="hubert_large_ll60k.pt"
#         url="https://dl.fbaipublicfiles.com/huggingface/pytorch-models/hubert/${hubert_model}"

#         if [ ! -f "${save_folder}/${hubert_model}" ]; then
#             echo "Downloading HuBERT model..."
#             curl -L "${url}" -o "${save_folder}/${hubert_model}"
#         else
#             echo "HuBERT model already exists. Skipping download."
#         fi
#         ;;
#     *)
#         echo "Invalid choice. Please select 1 or 2."
#         exit 1
#         ;;
# esac

# echo "Model setup complete."





#!/bin/bash
# Description: Download pretrained SSL models (wav2vec 2.0 / HuBERT)
# HuBERT is downloaded via Hugging Face Hub (official & reliable)

set -x

save_folder="./models"
mkdir -p "${save_folder}"

echo "Select the model you want to download:"
echo "1) wav2vec 2.0 (w2v_large_lv_fsh_swbd_cv)"
echo "2) HuBERT (facebook/hubert-large-ls960-ft)"
read -p "Enter your choice (1 or 2): " model_choice

case $model_choice in
    1)
        ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
        url="https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model}"

        if [ ! -f "${save_folder}/${ssl_model}" ]; then
            echo "Downloading wav2vec 2.0 model..."
            curl -L "${url}" -o "${save_folder}/${ssl_model}"
        else
            echo "wav2vec 2.0 model already exists. Skipping download."
        fi
        ;;

    2)
        echo "Downloading HuBERT via Hugging Face Hub..."

        # Ensure huggingface_hub is installed
        pip install -q huggingface_hub

        # Download model (weights + config)
        huggingface-cli download facebook/hubert-large-ls960-ft \
            --local-dir "${save_folder}/hubert-large-ls960-ft" \
            --local-dir-use-symlinks False
        ;;

    *)
        echo "Invalid choice. Please select 1 or 2."
        exit 1
        ;;
esac

echo "Model setup complete."


