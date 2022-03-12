import os 
import pandas as pd 
import numpy as np 

import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--metadata', type=str, default = "data/hupd_G06F1730_metadata_2022-03-04_bal.feather", help="Path to metadata.")
parser.add_argument('--min_n', type=int, default = 10, help="Minimum number of patents, otherwise will drop.")
parser.add_argument('--cv_groups', type=int, default = 5, help="CV groups to split into.")
parser.add_argument('--grp_col', type=str, default = "examiner_full_name", help="Variable to impute means on.")
args = parser.parse_args()

metadata = pd.read_feather(args.metadata)

def cv_lsom(df, grp_col, target_col, min_n=10):
	""" LEAVE SOME OUT MEAN (stupidly named but a lot simpler than a true LOOM)"""
    k = df['cv_group'].max()+1
    out_col = grp_col + "_" + target_col + "_mean"
    tmp_df = df.groupby([grp_col, "cv_group"])[target_col].agg(
        [(out_col, "sum"), ("c", "count")]
    ).reset_index()
    tmp_df = pd.concat( (tmp_df.assign(cv_main = i) for i in range(k) ))
    tmp_df = tmp_df.loc[ tmp_df['cv_main'] != tmp_df['cv_group'] ]
    res = tmp_df.groupby([grp_col, "cv_main"]).agg('sum')
    res[out_col] = res[out_col]/res['c']
    res = res.loc[res['c']>=min_n].drop(columns=['cv_group']).reset_index().rename(columns={'cv_main':'cv_group'})
    return res[[grp_col, "cv_group", out_col]]

# create necessary vars 
metadata = metadata.assign(
    decision_binary = lambda x: np.where(x['decision']=="ACCEPTED", 1, 0), 
    cv_group = np.random.choice( np.arange(0, args.cv_groups), size=metadata.shape[0] )
)

# calculate imputed means 
grp_lsom = cv_lsom(metadata, args.grp_col, "decision_binary", args.min_n)

# col names 
out_col = args.grp_col + "_decision_binary_mean"
cv_avg_out_col = "cv_avg" + out_col

# calculate cv average to fill in any NAS 
overall_cv_mean = grp_lsom.groupby(["cv_group"])[out_col].mean().rename(cv_avg_out_col).reset_index()

# merge all together 
metadata_lsom = metadata.merge(
	grp_lsom, on = [args.grp_col, "cv_group"], how = "left"
).merge(
	overall_cv_mean, on = ["cv_group"], how = "left"
)

# replace NAs w/ CV average 
metadata_lsom[out_col] = np.where(
		metadata_lsom[out_col].isna(), 
		metadata_lsom[cv_avg_out_col], metadata_lsom[out_col]
)

metadata_lsom = metadata_lsom.drop(columns = [cv_avg_out_col])

output_path = args.metadata.replace(".feather", "_lsom.feather")
metadata_lsom.to_feather(output_path)

