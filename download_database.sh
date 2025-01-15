set -x
	
FILE_NAMEs="LA"


for file in ${FILE_NAMEs}; do
    link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"

    if [ ! -d ./database/ASVspoof2019/${file} ] && [ ! -d ./database/${file}/flac ]; then
    # if [ ! -d ./database/ASVspoof2019/${file} ]; then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	echo ${link}
        wget -q --show-progress -c ${link} -O ${file}.zip
        # unzip -q ${file} -d LA
        UNZIP_FOLDER_PATH="./database/ASVspoof2019/"${file}""
        mkdir -p "$UNZIP_FOLDER_PATH"
        unzip -q ${file} -d "$UNZIP_FOLDER_PATH"
        rm ${file}.zip

    fi
done

echo 'We have ASVspoof2019 LA database now'



# #!/bin/bash
# set -x
	
# # FILE_NAMEs="train dev eval segment_labels_v1.2 protocols"
# FILE_NAMEs="train segment_labels_v1.2 dev protocols eval"
# # FILE_NAMEs="train dev segment_labels_v1.2 protocols"
# # FILE_NAMEs="eval"


# for file in ${FILE_NAMEs}; do

#     link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
#     if [ ! -d ./database/${file} ] && [ ! -d ./database/${file}/con_wav ]; then
#         echo -e "${RED}Downloading PartialSpoof ${name}"
# 	echo ${link}
#         wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
#         tar -zxvf database_${file}.tar.gz
#         rm database_${file}.tar.gz
#     fi
# done
# # remove any labels except for 0.64 resolution
# find /root/Partial_Spoof_Detection_System/database/segment_labels  -type f ! -name '*0.64*' -exec rm {} +
# echo 'We have PartialSpoof database now'