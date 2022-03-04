import os
import subprocess
import urllib.request 
import pandas as pd
import numpy as np 

# choose years 2005 - 2017
years = np.arange(2005, 2007, 1)
#years = np.arange(2005, 2018, 1)

def unpack_tar_gz(year): 
    ## change to data folder 
    os.chdir("data")

    ## execute bash comand
    bash_command = f"tar -zxvf data{year}.tar.gz"
    subprocess.run(bash_command, shell = True, check = True)

    ## change back to root 
    os.chdir('..')

## load in preferred nrs 
metadata = pd.read_feather("data/hupd_metadata_2022-02-22.feather")
sample_metadata = metadata.loc[metadata.main_ipcr_label == "G06F1730"]

id_nrs = sample_metadata['application_number']
json_nrs = id_nrs + ".json"

for year in years: 
    print("Editing year: ", year)
    # download file as tar gz
    dest_folder = f"data/data{year}.tar.gz"
    if not os.path.exists(dest_folder):
        huggingface_url = f"https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{year}.tar.gz"
        urllib.request.urlretrieve(huggingface_url, dest_folder)
    
    print("Retrieved hface data")

    ## unpack tar file
    if not os.path.isdir(f"data/{year}"): unpack_tar_gz(year)

    print("Unpacked tar")
    ## 
    json_files = os.listdir(f"data/{year}/")
    [os.remove(f"data/{year}/{file}") for file in json_files if file not in json_nrs]

    print("Removed undesirables ")
