set -x

ssl_model="w2v_large_lv_fsh_swbd_cv.pt"
save_folder="./models"  # Specify the folder here
if [ ! -e ${save_folder}/${ssl_model} ]; then
    wget --show-progress https://dl.fbaipublicfiles.com/fairseq/wav2vec/${ssl_model} -P ${save_folder}
fi