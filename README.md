# Identifying and Mitigating Spurious Correlations in Text Classification

This repository contains the code for our project, which focuses on identifying and mitigating spurious correlations in text classification tasks through causal inference and counterfactual methods. Our approach leverages a matching-based causal framework and applies counterfactual data augmentation to improve model robustness and fairness.



We conduct experiments on four diverse datasets:
- **IMDB Movie Reviews**: A classic sentiment analysis dataset where movie review sentences are labeled as positive or negative.  
- **Kindle Reviews**: Product review snippets from the Amazon Kindle Store, with ratings converted into binary sentiment labels (1–2 as negative and 4–5 as positive).
- **Toxic Comment**: Wikipedia talk-page comments, where toxicity is assessed based on crowd-sourced annotations.
- **Toxic Tweet**: Tweets collected via the Twitter Streaming API and labeled as toxic or non-toxic based on human ratings.

> **Note:** Since our pickle files used during data preprocessing and embedding steps are larger than 2 GB, GitHub does not allow them to be uploaded. Please refer to the scripts in the repository that describe how these pickle files are created. Once created, these files are read during various processing steps to speed up subsequent experiments.you can refer on how i've created the pickle files and use those variables to run the scripts, , that's for entire project however, you can test the code with project_demo with_imdb  and project_test_with_imdb to understand and check the code , since this is ran only on imdb dataset, it should not take long except for vectorization tasks 


#### Datasets summary
 
| Dataset  | #docs | top terms (coef>=1) | #matched sentences for top terms | placebo terms | #matched sentences for placebo terms |
| ---------| ------| --------------------| ---------------------------------| --------------| -------------------------------------|
| IMDB | 10,662 | 366 | 8,882 | 626 (coef<=0.1) | 12,996 |
| Kindle | 20,233 (N, 10,161) (P, 10,072) | 270 | 24,882 | 569 (coef <=0.2) | 13,850|
| Toxic comment | 15,216 | 329 | 8,414 | 750 (coef<=0.1) | 30,454 |
| Toxic tweet | 6,774 | 341 (coef >=0.7) | 9,224 | 574 (coef<=0.2) | 5,457 |



## Project Overview

The key steps in our pipeline include:
- **Initial Feature Extraction:** Training a bag-of-words logistic regression classifier to identify influential words.
- **Context Editing and Embedding:** Editing sentences to remove candidate words and generating context-rich representations using BERT.
- **Matching and Causal Effect Computation:** Identifying control sentences via cosine similarity and computing the Average Treatment Effect (ATE) for each word.
- **Word-Level Supervised Classification:** Training a lightweight classifier on manually annotated words to predict the spuriousness of each candidate.
- **Counterfactual Data Augmentation:** Generating augmented training samples by replacing spurious words with appropriate synonyms.
- **Domain Adaptation and Robustness Evaluation:** Analyzing subgroup performance (majority vs. minority) and evaluating cross-domain transferability.

### Please refer the video here for demo : https://drive.google.com/file/d/1Xh3f2AM6sLGIvAgsftzVnWc1Npv0OTPa/view?usp=sharing

### Create a virtual environment (optional but recommended):
python -m venv venv
> **note** - you need to refer your python path here 
source venv/bin/activate 
On Windows: venv\Scripts\activate


### Install dependencies
All the required libraries are listed in the requirements.txt file.

To install them, run:
pip install -r requirements.txt

make sure you run the above command in your virtual enviromnent

> **note** - Please be aware that first import will take long to import the modules after installation
