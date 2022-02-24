import os
import urllib.request
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--year', type=str, required=True, help="Year of data to fetch; format XXXX.")

args = parser.parse_args()

assert args.year.isdigit(), "That's not a number."
assert int(args.year) >= 2004 and int(args.year) <= 2018, "We don't have data for that year."

urllib.request.urlretrieve(f"https://huggingface.co/datasets/HUPD/hupd/resolve/main/data/{args.year}.tar.gz", f"data{args.year}.tar.gz")
os.rename(f"data{args.year}.tar.gz", f"../../data/data{args.year}.tar.gz")
