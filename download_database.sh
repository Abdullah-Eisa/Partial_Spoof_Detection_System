#!/bin/bash
set -x
	
FILE_NAMEs="train dev eval segment_labels_v1.2 protocols"


for file in ${FILE_NAMEs}; do

    link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
    if [ ! -d ./database/${file} ] && [ ! -d ./database/${file}/con_wav ]; then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	echo ${link}
        wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
        tar -zxvf database_${file}.tar.gz
        rm database_${file}.tar.gz
    fi
done
# remove any labels except for 0.64 resolution
find /root/Partial_Spoof_Detection_System/database/segment_labels  -type f ! -name '*0.64*' -exec rm {} +
echo 'We have PartialSpoof database now'