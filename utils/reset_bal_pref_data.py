import os
import shutil 
import pandas as pd 
import argparse 

os.chdir('data')

parser = argparse.ArgumentParser()
parser.add_argument('--ipc_code', type=str, default = "G06F1730", help="IPC Code to select: default G06F1730.")

# load sample metadata
bal_sample_metadata = pd.read_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04_bal.feather")
bal_json_files = bal_sample_metadata['application_number'] + ".json"
bal_jf_year = bal_sample_metadata['filing_date'].dt.year

jf_to_year = dict(zip(bal_json_files, bal_jf_year))

files_in_balanced_folder = os.listdir("bal_data")

for jf in files_in_balanced_folder:
	# get associated year 
	jf_yr = jf_to_year[jf]
	# place in that location 
	shutil.move("bal_data/" + jf, f"{jf_yr}/" + jf)
