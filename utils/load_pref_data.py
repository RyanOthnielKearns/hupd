import os
import subprocess
import shutil
import urllib.request 
import pandas as pd
import numpy as np 
from tqdm import tqdm 

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ipc_code', type=str, default = "G06F1730", help="IPC Code to select: default G06F1730.")
parser.add_argument('--min_year', type=int, default = "2004", help="Year to start range at: default 2004.")
parser.add_argument('--max_year', type=int, default = "2017", help="Year to end range at: default 2017.")

args = parser.parse_args()

print("Running with IPC code:", args.ipc_code)

if args.min_year < 2004: 
    raise Exception("Min year must be >= 2004.")
elif args.max_year > 2018:
    raise Exception("Max year must be <= 2018.")
elif args.min_year > args.max_year:
    raise Exception("Max year cannot be less than min year.")

# choose years 2005 - 2017
years = np.arange(args.min_year, args.max_year + 1, 1)

## change to data dir 
os.chdir('data')

## load in selected metadata 
if not os.path.exists(f"hupd_{args.ipc_code}_metadata_2022-03-04.feather"):
    metadata = pd.read_feather("hupd_metadata_2022-02-22.feather")
    sample_metadata = metadata.loc[metadata.main_ipcr_label == args.ipc_code]
    if sample_metadata.shape[0] == 0:
        raise Exception("Invalid IPC Code. Metadata has zero rows.")
    sample_metadata = sample_metadata.reset_index(drop=True)
    sample_metadata.to_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04.feather")
else: 
    sample_metadata = pd.read_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04.feather")

## get json files for this category 
app_files = sample_metadata['application_number'] + ".json"

def retrieve_data(year): 
    # download file as tar gz
    dest_folder = f"data{year}.tar.gz"
    if not os.path.exists(dest_folder):
        huggingface_url = f"https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{year}.tar.gz"
        urllib.request.urlretrieve(huggingface_url, dest_folder)

def unpack_tar_gz(year): 
    ## execute bash comand
    bash_command = f"tar -zxvf data{year}.tar.gz"
    subprocess.run(bash_command, shell = True, check = True)


def select_desired_files(year, app_files): 
    # get json files from tar 
    json_files = os.listdir(f"{year}/")

    # keep all files in intersection of app files + json files 
    app_files_curyear = set(app_files).intersection(set(json_files))

    # delete all files not in app_files 
    files_to_delete = set(json_files) - set(app_files)

    # print out some stats 
    print("Files to delete:", len(list(files_to_delete)))
    print("Files to keep:", len(list(app_files_curyear)))

    # create new dir to store files 
    new_dir_path = f"seldata{year}"
    if not os.path.isdir(new_dir_path): os.mkdir(new_dir_path)

    # move files (ie check if non empty --> )
    if len(os.listdir(new_dir_path)) == 0:
        for af in app_files_curyear:
            shutil.move(f"{year}/{af}", new_dir_path + "/" + af)

    # remove other files:
    shutil.rmtree(f"{year}")

    # remove tar files
    os.remove(f"data{year}.tar.gz")




for year in tqdm(years): 

    print("Operating on year", year)

    retrieve_data(year)
    
    print("Retrieved data.")

    ## unpack tar file
    if not os.path.isdir(f"{year}"): unpack_tar_gz(year)

    print("Opened tar file.")
    
    select_desired_files(year, app_files)
    print("Removed undesirables ")
