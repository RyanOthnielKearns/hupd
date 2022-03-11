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
parser.add_argument('--min_year', type=int, default = "2005", help="Year to start range at: default 2004.")
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

#all_json_ids = set()
#for year in np.arange(2005, 2017 + 1, 1):
#	all_json_ids.add(set([ i.split(".")[0] for i in os.listdir(f"seldata{year}/") ]))

sample_metadata = pd.read_feather(f"hupd_{args.ipc_code}_metadata_2022-03-04.feather")

## filter to desired year range 
sample_metadata = sample_metadata.loc[
	sample_metadata['filing_date'].dt.year.isin(years)
]

# filter to just accepted + rejected 
sample_metadata = sample_metadata.loc[
	np.logical_or(
		sample_metadata['decision'] == "ACCEPTED",
		sample_metadata['decision'] == "REJECTED",
	) 
	#& sample_metadata['application_number'] in all_json_ids
].assign(
	decision_binary = lambda x: np.where(x['decision'] == "ACCEPTED", 1, 0)
)

# get true statistics 
true_N = sample_metadata['decision_binary'].shape[0]
true_N_accepted = sample_metadata['decision_binary'].sum()
true_N_rejected=  true_N - true_N_accepted
true_data_prop = true_N_accepted / (true_N)

print("Actual split: ")
print(f"{true_N_accepted} actual accepted patents")
print(f"{true_N_rejected} actual rejected patents\n")

# get desired statistics 
desired_N_accepted = int(args.bal_prop * true_N)
desired_N_rejected = true_N - desired_N_accepted

print("Desired split: ")
print(f"{desired_N_accepted} desired accepted patents")
print(f"{desired_N_rejected} desired rejected patents\n")

# note if we don't have enough for true split
if desired_N_rejected > true_N_rejected:
	desired_N_rejected = true_N_rejected
	desired_N_accepted = int(desired_N_rejected * (args.bal_prop / (1 - args.bal_prop)))
if desired_N_accepted > true_N_accepted:
	desired_N_accepted = true_N_accepted
	desired_N_rejected = int(desired_N_accepted * ((1 - args.bal_prop) / args.bal_prop))

#print(f"Selecting {desired_N_accepted} accepted patents and {desired_N_rejected} rejected patents to fix balance of {args.bal_prop}")

print("Required split for proper balance:")
print(f"{desired_N_accepted} accepted patents")
print(f"{desired_N_rejected} rejected patents")

## filter to balanced sample 
sample_metadata_acc = sample_metadata.loc[
	sample_metadata['decision_binary'] == 1
].reset_index().sample(n = desired_N_accepted, replace = False, random_state = args.seed)

sample_metadata_rej = sample_metadata.loc[
	sample_metadata['decision_binary'] == 0
].reset_index().sample(n = desired_N_rejected, replace = False, random_state = args.seed)

## combine together 
bal_sample_metadata = pd.concat((sample_metadata_acc, sample_metadata_rej), axis = 0).reset_index()

## save
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
	files_in_balanced_sample = set(os.listdir(f"seldata{year}/")).intersection(set(bal_json_files))
	for jf in files_in_balanced_sample:
		shutil.move(f"seldata{year}/" + jf, "bal_data/" + jf)

