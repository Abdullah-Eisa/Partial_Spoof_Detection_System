#!/bin/bash
set -x

# Prompt the user to choose a database
echo "Select the database you want to download:"
echo "1) RFP_database"
echo "2) PartialSpoof"
echo "3) ASVspoof2019 LA"
read -p "Enter the number of your choice: " db_choice

case $db_choice in
    1)
        echo "You chose to download the RFP_database."

        FILE_NAMEs="RFP_database"
        for file in ${FILE_NAMEs}; do
            link="https://zenodo.org/records/14675126/files/database.zip?download=1"

            if [ ! -d ./database/RFP_database/ ]; then
                echo -e "${RED}Downloading RFP_database ${file}"
                echo ${link}
                wget -q --show-progress -c ${link} -O RFP_database.zip
                UNZIP_FOLDER_PATH="./database/"
                mkdir -p "$UNZIP_FOLDER_PATH"
                unzip -q RFP_database.zip -d "$UNZIP_FOLDER_PATH"
                rm RFP_database.zip

                mkdir -p ./database/RFP/
                mv ./database/database/* ./database/RFP/
                rm -r ./database/database
            fi
        done

        echo 'We have RFP database now'
        ;;

    2)
        echo "You chose to download the PartialSpoof database."
        # code adapted from: https://github.com/nii-yamagishilab/PartialSpoof/blob/847347aaec6f65c3c6d2f17c63515b826b94feb3/01_download_database.sh
        FILE_NAMEs="train segment_labels_v1.2 dev protocols eval"
        TARGET_DIR="./database/PartialSpoof"  # Specify your target directory here

        # Create the target directory if it doesn't exist
        mkdir -p $TARGET_DIR

        for file in ${FILE_NAMEs}; do
            link="https://zenodo.org/record/5766198/files/database_"${file}".tar.gz?download=1"
            if [ ! -d $TARGET_DIR/${file} ] && [ ! -d $TARGET_DIR/${file}/con_wav ]; then
                echo -e "${RED}Downloading PartialSpoof ${file}"
                echo ${link}
                wget -q --show-progress -c ${link} -O  database_${file}.tar.gz
                
                # Extract the tar.gz file to the specified directory
                tar -zxvf database_${file}.tar.gz -C $TARGET_DIR
                
                # Clean up the tar.gz file after extraction
                rm database_${file}.tar.gz
            fi
        done
        # remove any labels except for 0.64 resolution
        find /root/Partial_Spoof_Detection_System/database/segment_labels  -type f ! -name '*0.64*' -exec rm {} +

        echo 'We have PartialSpoof database now'
        ;;

    3)
        echo "You chose to download the ASVspoof2019 LA database."

        FILE_NAMEs="LA"
        for file in ${FILE_NAMEs}; do
            link="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip?sequence=3&isAllowed=y"

            if [ ! -d ./database/ASVspoof2019/${file} ] && [ ! -d ./database/${file}/flac ]; then
                echo -e "${RED}Downloading ASVspoof2019 LA ${file}"
                echo ${link}
                wget -q --show-progress -c ${link} -O ${file}.zip
                UNZIP_FOLDER_PATH="./database/ASVspoof2019/"
                mkdir -p "$UNZIP_FOLDER_PATH"
                unzip -q ${file} -d "$UNZIP_FOLDER_PATH"
                rm ${file}.zip
            fi
        done

        echo 'We have ASVspoof2019 LA database now'
        ;;

    *)
        echo "Invalid choice. Please select a valid option (1, 2, or 3)."
        ;;
esac
