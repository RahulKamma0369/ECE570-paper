# Identifying and Mitigating Spurious Correlations in Text Classification

This repository contains the code for our project, which focuses on identifying and mitigating spurious correlations in text classification tasks through causal inference and counterfactual methods. Our approach leverages a matching-based causal framework and applies counterfactual data augmentation to improve model robustness and fairness.

We conduct experiments on four diverse datasets:
- **IMDB Movie Reviews**: A classic sentiment analysis dataset where movie review sentences are labeled as positive or negative.  
- **Kindle Reviews**: Product review snippets from the Amazon Kindle Store, with ratings converted into binary sentiment labels (1–2 as negative and 4–5 as positive).
- **Toxic Comment**: Wikipedia talk-page comments, where toxicity is assessed based on crowd-sourced annotations.
- **Toxic Tweet**: Tweets collected via the Twitter Streaming API and labeled as toxic or non-toxic based on human ratings.



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


### Create a virtual environment (optional but recommended):
python -m venv venv

source venv/bin/activate 
On Windows: venv\Scripts\activate


### Install dependencies
All the required libraries are listed in the requirements.txt file.
To install them, run:
pip install -r requirements.txt
