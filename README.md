![HUPD-Diagram](https://github.com/suzgunmirac/hupd-test/blob/main/figures/HUPD-Logo.png)

# The Harvard USPTO Patent Dataset (HUPD)
This present repository contains the dataset from "[_The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications_](https://openreview.net/pdf?id=WhTTCWsMrYv)", which is currently under review in the NeurIPS 2021 Datasets and Benchmarks Track.

## Table of Contents
1. [Overview of HUPD](#overview-of-hupd)
    * [Downloading the Dataset](#downloading-the-dataset)
    * [Data Fields and Data Format](#data-fields-and-data-format)
    * [Jupyter Notebooks](#jupyter-notebooks)
2. [Experiments and Tasks](#experiments-and-tasks)
    * [Model Weights](#model-weights)
3. [Citation](#citation)
4. [Licensing and Disclaimer](#licensing-and-disclaimer)
5. [Contact and Contributing](#contact-and-contributing)

## Overview of HUPD
The Harvard USPTO Patent Dataset (HUPD) is a large-scale, well-structured, and multi-purpose corpus of English-language patent applications filed to the United States Patent and Trademark Office (USPTO) between 2004 and 2018. With more than 4.5 million patent documents, HUPD is two to three times larger than comparable corpora. Unlike other NLP patent datasets, HUPD contains the inventor-submitted versions of patent applications, not the final versions of granted patents, allowing us to study patentability at the time of filing using NLP methods for the first time.

HUPD distinguishes itself from prior patent datasets in three key aspects: 
- (1) Unlike previous datasets, it focuses on patent applications, not merely granted patents, since patent applications contain the original set of claims and descriptions of the proposed invention written by the applicant. In fact, having access to the original versions of both accepted and rejected applications allows us to introduce a completely new task to the field – namely, the binary classification of patent decisions, wherein the goal is to predict the acceptance likelihood of a patent application at the time of submission.
- (2) It is the first NLP dataset to contain multiple classes of rich textual and structural information present in patent applications. Whereas previous datasets include only one or two of a patent’s data fields (e.g., description and abstract), HUPD contains 34 fields, including filing date, fine-grained classification codes, examiner information, and many others. The variety of information available for each patent application can enable NLP researchers to perform a wide range of tasks—such as analyzing the evolution of patent language and categories over time—that were not possible under previous patent datasets.
- (3) HUPD uses information obtained directly from the USPTO, rather than from Google’s Patent search; it is significantly larger than previous datasets while still being clean, comprehensive, and well-structured.

### Downloading the Dataset
The dataset is a available via four large tarred files and a big feather file. **More than 360GB of storage space** is needed to download and store all the individual files.

The following command will download all of them and extract the archives:

```bash
bash ./scripts/download_and_extract_all.sh
```

### Data Fields and Data Format
Each patent application is defined by a distinct JSON file, named after its application number, and includes information about the application and publication numbers, title, decision status, filing and publication dates, primary and secondary classification codes, inventor(s), examiner, attorney, abstract, claims, background, summary, and full description of the proposed invention, among other fields. There are also supplementary variables, such as the small-entity indicator (which denotes whether the applicant is considered to be a small entity by the USPTO) and the foreign-filing indicator (which denotes whether the application was originally filed in a foreign country). 

- In total, there are 34 data fields for each application.

### Jupyter Notebooks 
Please feel free to take a look at our notebooks if you would like to run the code in an interactive session or plot some of the figures in our paper by yourself.
* `Exploring the Data Fields of HUPD.ipynb`: To explore some of the data fields within HUPD.
* `Loading HUPD By Using HuggingFace's Libraries.ipynb`: To learn how to load and use HUPD using Hugging Face's libraries. 

### Google Colab
Here is a [Google Colab notebook](https://colab.research.google.com/drive/1cIS5aftnp6Azvqiiqv6w-AnKpopYSCUZ?usp=sharing) that illustrates how to use our dataset.

## Experiments and Tasks
Let us first provide a brief overview of each task we consider in our paper:
- **Patent Acceptance Prediction**: Given a section of a patent application (in particular, the asbtract, claims, or description), we predict whether the application will be accepted by the USPTO.
- **Automated Subject (IPC/CPC) Classification**: We predict the primary IPC or CPC code  of a patent application given (some subset of) the text of the application.
- **Language Modeling**: We perform masked language modeling on the claims and description sections of patent applications.
- **Abstractive Summarization**: Each patent contains an abstract section in which the applicant summarizes the content of the patent. We use this section as the ground truth for our abstractive summarization task, and we use either the claims section or the description section as the source text.

### Model Weights
The model weights can be downloaded from [this Google Drive link](https://drive.google.com/drive/folders/12c6tIsaKisTR-ujukGXjbllk6gRGFRvx?usp=sharing).

## Citation
If your research makes use of our dataset or our empirical findings, please consider citing our work. 
```
@article{suzgun2021hupd,
  title={The Harvard USPTO Patent Dataset: A Large-Scale, Well-Structured, and Multi-Purpose Corpus of Patent Applications},
  author={Suzgun, Mirac and Sarkar, Suproteem K and Melas-Kyriazi, Luke and Kominers, Scott and Shieber, Stuart},
  year={2021}
}
```

## Licensing and Disclaimer
- This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License. 
- Any opinions, findings, and conclusions expressed in our paper are those of the authors and do not necessarily reflect the views of Harvard University or the USPTO. All errors remain our own.

## Contact and Contributing
- Please feel free to make use of the "Issues" page to ask your dataset-specific questions or to hold a public discussion.
- We look forward to seeing what the NLP and IP communities do with our publicly available dataset and models.