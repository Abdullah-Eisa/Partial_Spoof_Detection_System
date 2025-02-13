# Description: Download the pre-trained wav2vec 2.0 model from the official fairseq repository
# code adapted from: https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/03multireso/01_download_pretrained_models.sh
set -x

ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
save_folder="./models"  # Specify the folder here
if [ ! -e ${save_folder}/${ssl_model} ]; then
    wget --show-progress https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model} -P ${save_folder}
fi