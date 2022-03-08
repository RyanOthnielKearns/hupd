import os 
import urllib.request


metadata_path = "https://huggingface.co/datasets/HUPD/hupd/resolve/main/hupd_metadata_2022-02-22.feather"
local_path = "data/hupd_metadata_2022-02-22.feather"

urllib.request.urlretrieve(metadata_path, local_path)