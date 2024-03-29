"""
The Harvard USPTO Patent Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus 
of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) 
between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger 
than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions 
of patent applications, not the final versions of granted patents, allowing us to study patentability at 
the time of filing using NLP methods for the first time.
"""

from __future__ import absolute_import, division, print_function

import os
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
try:
    import ujson as json
except:
    import json

import datasets


_CITATION = """\
@InProceedings{suzgun2021:hupd,
title = {The Harvard USPTO Patent Dataset},
authors={Mirac Suzgun and Suproteem Sarkar and Luke Melas-Kyriazi and Scott Kominers and Stuart Shieber},
year={2021}
}
"""

_DESCRIPTION = """
The Harvard USPTO Patent Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus 
of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) 
between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger 
than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions 
of patent applications, not the final versions of granted patents, allowing us to study patentability at 
the time of filing using NLP methods for the first time.
"""

RANDOM_STATE = 1729

_FEATURES = [
    "patent_number",
    "decision",
    "title",
    "abstract",
    "claims",
    "background",
    "summary",
    "description",
    "cpc_label",
    "ipc_label",
    "filing_date",
    "patent_issue_date",
    "date_published",
    "examiner_id", 
    "examiner_id_impute_mean",
    "patent_year"
]


def str_to_date(s):
    """A helper function to convert strings to dates"""
    return datetime.datetime.strptime(s, '%Y-%m-%d')


class PatentsConfig(datasets.BuilderConfig):
    """BuilderConfig for Patents"""

    def __init__(
        self,
        metadata_file: str,
        data_dir: str,
        ipcr_label: str = None,
        cpc_label: str = None,
        train_filing_start_date: str = None,
        train_filing_end_date: str = None,
        val_filing_start_date: str = None,
        val_filing_end_date: str = None,
        query_string: str = None,
        val_set_balancer=False,
        uniform_split=False,
        **kwargs
    ):
        """
        If train_filing_end_date is None, then a random train-val split will be used. If it is 
        specified, then the specified date range will be used for the split. If train_filing_end_date 
        if specified and val_filing_start_date is not specifed, then val_filing_start_date defaults to 
        train_filing_end_date. 

        Args:
            metadata_file: `string`, the metadata file
            data_dir: `string`, folder (in cache) in which downloaded json files are stored
            ipcr_label: International Patent Classification code
            cpc_label: Cooperative Patent Classification code
            train_filing_start_date: Start date for patents in train set (and val set if random split is used)
            train_filing_end_date: End date for patents in train set
            val_filing_start_date: Start date for patents in val set
            val_filing_end_date: End date for patents in val set (and train set if random split is used)
            **kwargs: keyword arguments forwarded to super
        """
        super().__init__(**kwargs)
        self.metadata_file = metadata_file
        self.data_dir = data_dir
        self.ipcr_label = ipcr_label
        self.cpc_label = cpc_label
        self.train_filing_start_date = train_filing_start_date
        self.train_filing_end_date = train_filing_end_date
        self.val_filing_start_date = val_filing_start_date
        self.val_filing_end_date = val_filing_end_date
        self.query_string = query_string
        self.val_set_balancer = val_set_balancer
        self.uniform_split = uniform_split


class Patents(datasets.GeneratorBasedBuilder):
    _DESCRIPTION

    VERSION = datasets.Version("1.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.
    BUILDER_CONFIG_CLASS = PatentsConfig
    BUILDER_CONFIGS = [
        PatentsConfig(
            name="sample", 
            description="Patent data from January 2016, for debugging", 
            metadata_file="../data/hupd_G06F1730_metadata_2022-03-04_bal.feather",
            data_dir="../data/bal_data/"
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {k: datasets.Value("string") for k in _FEATURES}
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=("claims", "decision"),
            homepage="https://github.com/suzgunmirac/hupd",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        """Returns SplitGenerators."""
        print(f'Loading dataset with config: {self.config}')

        # Download metadata
        # NOTE: Metadata is stored as a Pandas DataFrame in Apache Feather format
        metadata_file = self.config.metadata_file
        print(f'Using metadata file: {metadata_file}')

        # Download data
        # NOTE: The extracted path contains a subfolder, data_dir. This directory holds
        # a large number of json files (one json file per patent application).
        json_dir = os.path.join(self.config.data_dir)

        # Load metadata file
        df = pd.read_feather(metadata_file)

        ## add patent_year col 
        df = df.assign(
            patent_year = lambda x: x['filing_date'].dt.year
        )

        # Filter based on ICPR / CPC label
        if self.config.ipcr_label:
            print(f'Filtering by IPCR label: {self.config.ipcr_label}')
            df = df[df['main_ipcr_label'].str.startswith(self.config.ipcr_label)]
        elif self.config.cpc_label:
            print(f'Filtering by CPC label: {self.config.cpc_label}')
            df = df[df['main_cpc_label'].str.startswith(self.config.cpc_label)]

        # Filter metadata based on arbitrary query string
        if self.config.query_string:
            df = df.query(self.config.query_string)

        # Train-validation split (either uniform or by date)
        if self.config.uniform_split:

            # Assumes that training_start_data < val_end_date
            if self.config.train_filing_start_date:
                df = df[df['filing_date'] >= self.config.train_filing_start_date]
            if self.config.val_filing_end_date:
                df = df[df['filing_date'] <= self.config.val_filing_end_date]
            df = df.sample(frac=1.0, random_state=RANDOM_STATE)
            num_train_samples = int(len(df) * 0.85)
            train_df = df.iloc[0:num_train_samples]
            val_df = df.iloc[num_train_samples:-1]

        else:

            # Check
            if not (self.config.train_filing_start_date and self.config.train_filing_end_date and
                    self.config.val_filing_start_date and self.config.train_filing_end_date):
                raise ValueError("Please either use uniform_split or specify your exact \
                    training and validation split dates.")

            # Does not assume that training_start_data < val_end_date
            print(f'Filtering train dataset by filing start date: {self.config.train_filing_start_date}')
            print(f'Filtering train dataset by filing end date: {self.config.train_filing_end_date}')
            print(f'Filtering val dataset by filing start date: {self.config.val_filing_start_date}')
            print(f'Filtering val dataset by filing end date: {self.config.val_filing_end_date}')
            train_df = df[
                (df['filing_date'] >= self.config.train_filing_start_date) & 
                (df['filing_date'] < self.config.train_filing_end_date)
            ]
            val_df = df[
                (df['filing_date'] >= self.config.val_filing_start_date) & 
                (df['filing_date'] < self.config.val_filing_end_date)
            ]

        # TODO: We can probably make this step faster
        if self.config.val_set_balancer:
            rejected_df = val_df[val_df.decision == 'REJECTED']
            num_rejected = len(rejected_df)
            accepted_df = val_df[val_df.decision == 'ACCEPTED']
            num_accepted = len(accepted_df)
            if num_rejected < num_accepted:
                accepted_df = accepted_df.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle(accepted_df)
                accepted_df = accepted_df[:num_rejected]
            else:
                rejected_df = rejected_df.sample(frac=1.0, random_state=RANDOM_STATE)  # shuffle(rejected_df)
                rejected_df = rejected_df[:num_accepted]
            val_df = pd.concat([rejected_df, accepted_df])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs=dict(  # these kwargs are passed to _generate_examples
                    df=train_df,
                    json_dir=json_dir,
                    split='train',
                ),
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs=dict(
                    df=val_df,
                    json_dir=json_dir,
                    split='val',
                ),
            ),
        ]

    def _generate_examples(self, df, json_dir, split):
        """ Yields examples by loading JSON files containing patent applications. """

        # NOTE: df.itertuples() is way faster than df.iterrows()
        for id_, x in enumerate(df.itertuples()):

            # JSON files are named by application number (unique)
            application_number = x.application_number
            filepath = os.path.join(json_dir, application_number + '.json')
            try:
                with open(filepath, 'r') as f:
                    patent = json.load(f)
            except Exception as e:
                print('------------')
                print(f'ERROR WITH {filepath}\n')
                print(repr(e))
                print()
                yield id_, {k: "error" for k in _FEATURES}

            # Most up-to-date-decision in meta dataframe
            decision = x.decision
            examiner_id_impute_mean = x.examiner_id_impute_mean
            patent["patent_year"] = str(x.patent_year)

            yield id_, {
                "patent_number": application_number,
                "decision": decision,
                "title": patent["title"],
                "abstract": patent["abstract"],
                "claims": patent["claims"],
                "description": patent["full_description"],
                "background": patent["background"],
                "summary": patent["summary"],
                "cpc_label": patent["main_cpc_label"],
                'filing_date': patent['filing_date'],
                'patent_issue_date': patent['patent_issue_date'],
                'date_published': patent['date_published'],
                'examiner_id': patent["examiner_id"] if not patent["examiner_id"] == "" else "0",
                "examiner_id_impute_mean": examiner_id_impute_mean if not None else 0.5,
                "ipc_label": patent["main_ipcr_label"],
                "patent_year": patent["patent_year"] if not patent["patent_year"] == "" else "0"
            }
