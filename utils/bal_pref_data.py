import os 
import shutil
import pandas as pd 
import numpy as np
import argparse 

os.chdir('data')

parser = argparse.ArgumentParser()
parser.add_argument('--ipc_code', type=str, default = "G06F1730", help="IPC Code to select: default G06F1730.")
parser.add_argument('--bal_prop', type=float, default = 0.50, help="Balance proportion (out of 1): higher number --> more accepted.")
parser.add_argument('--seed', type=int, default = 224, help = "Set the random seed for split reproducibility.")

sample_metadata = pd.read_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04.feather")

sample_metadata = sample_metadata.loc[
	np.logical_or(
		sample_metadata['decision'] == "ACCEPTED",
		sample_metadata['decision'] == "REJECTED",
	)
].assign(
	decision_binary = lambda x: np.where(x['decision'] == ACCEPTED, 1, 0)
)

true_N = sample_metadata['decision_binary'].shape[0]
true_N_accepted = sample_metadata['decision_binary'].sum()
true_N_rejected=  true_N - true_N_accepted
true_data_prop = true_N_accepted / (true_N)


desired_N_accepted = int(parser.bal_prop * true_N)
desired_N_rejected = true_N - desired_N_accepted

# note if we don't have enough for true split
if desired_N_rejected > true_N_rejected:
	desired_N_rejected = true_N_rejected
	desired_N_accepted = int(desired_N_rejected * (parser.bal_prop / (1 - parser.bal_prop)))
else if desired_N_accepted > true_N_accepted: 
	desired_N_accepted = true_N_accepted
	desired_N_rejected = int(desired_N_accepted * ((1 - parser.bal_prop) / parser.bal_prop))

sample_metadata_acc = sample_metadata.loc[
	sample_metadata['decision_binary'] == 1
].reset_index().sample(n = desired_N_accepted, replace = False, random_state = parser.seed)

sample_metadata_rej = sample_metadata.loc[
	sample_metadata['decision_binary'] == 0
].reset_index().sample(n = desired_N_rejected, replace = False, random_state = parser.seed)

bal_sample_metadata = pd.concat((sample_metadata_acc, sample_metadata_rej), axis = 0)

bal_sample_metadata.to_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04_bal.feather")

# construct balanced data
bal_json_files = bal_sample_metadata['application_number'] + ".json"

year_min = bal_sample_metadata['filing_date'].dt.year.min()
year_max = bal_sample_metadata['filing_date'].dt.year.max()
year_range = np.arange(year_min, year_max + 1, 1)

## move files to balanced data folder

if not os.path.exists('bal_data'):
	os.mkdir('bal_data')

for year in year_range:
	files_in_balanced_sample = set(os.listdir(f"{year}/")).intersection(set(bal_json_files))
	for jf in files_in_balanced_sample:
		shutil.move(f"{year}/" + jf, "bal_data/" + jf)






