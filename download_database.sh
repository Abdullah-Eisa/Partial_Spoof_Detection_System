set -x
	
# FILE_NAMEs="train dev eval segment_labels_v1.2 protocols"
# FILE_NAMEs="train segment_labels_v1.2 dev protocols eval"
# FILE_NAMEs="train dev segment_labels_v1.2 protocols"
FILE_NAMEs="README.txt"


for file in ${FILE_NAMEs}; do

    # link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/asvspoof2019_evaluation_plan.pdf?sequence=1&isAllowed=y"
    link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/README.txt?sequence=5&isAllowed=y"
    # if [ ! -d ./database/ASVspoof2019/${file} ] && [ ! -d ./database/${file}/con_wav ]; then
    if [ ! -d ./database/ASVspoof2019/${file} ]; then
        echo -e "${RED}Downloading PartialSpoof ${name}"
	# echo ${link}
        wget -q --show-progress -c ${link} -O  README.txt
        # tar -zxvf database_${file}.tar.gz
        # rm database_${file}.tar.gz
    fi
done
# remove any labels except for 0.64 resolution
echo 'We have PartialSpoof database now'




# #!/bin/bash
# set -x

# # Define the file names and links to download
# FILE_NAMEs="asvspoof2019_evaluation_plan.pdf LA.zip"

# for file in ${FILE_NAMEs}; do
#     # Define the URL based on the file name
#     if [ "${file}" == "asvspoof2019_evaluation_plan.pdf" ]; then
#         link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/asvspoof2019_evaluation_plan.pdf?sequence=1&isAllowed=y"
#     elif [ "${file}" == "LA.zip" ]; then
#         link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"
#     fi

#     # Check if the file exists before downloading
#     if [ ! -f ./${file} ]; then
#         echo -e "Downloading ${file}"
#         wget -q --show-progress -c ${link} -O ${file}
#     else
#         echo -e "${file} already exists, skipping download."
#     fi
# done

# echo "Download complete!"
