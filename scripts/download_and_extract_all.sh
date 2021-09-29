#!/bin/sh

# Google Drive file ids 
ids=('19z-hhIWkHXqyx-W2t3gc5ZerFdORQ6vP' '19z-13vmHfR5XCO13sctypRpGtamqeYqe-nRX' '19z-1HIj7i5oDVFyQHbvQVCa-7SJAxbjceIR1' '19z-1tAPUX0CrEJ2PNNUW7Xc2OQXzPJooobOj')

# Install all the tar files via gdown
install_tar_files() {
    for id in ${ids[@]}; do
        echo Downloading https://drive.google.com/uc?id=${id}
        gdown https://drive.google.com/uc?id=${id}
    done
}

# Extract all the tar files in the directory at once
untar_tar_files() {
    tar -zxvf *.tar.gz.*
}

install_tar_files
untar_tar_files