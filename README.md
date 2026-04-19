# Sentiment Analysis of Amazon Product Reviews
 
**Comparing Traditional and Modern Text Mining Approaches**
 
MSc in Economics with Data Science — University of Alicante
Text Mining Module — Final Project
 
## Overview
 
This project performs binary sentiment classification on Amazon food reviews, comparing traditional text mining representations (TF, TF-IDF, n-grams) against modern BERT-based sentence embeddings. The pipeline covers exploratory linguistic analysis, text preprocessing, feature extraction, classification, and error analysis.
 
## Key Results
 
| Representation | Best Classifier | Macro F1 |
|---|---|---|
| TF-IDF Uni+Bi+Trigrams | Linear SVM | **0.871** |
| TF-IDF Uni+Bigrams | Linear SVM | 0.867 |
| TF Uni+Bigrams | Logistic Regression | 0.864 |
| Sentence Embeddings (BERT) | Logistic Regression | 0.798 |
| Baseline (majority class) | — | 0.333 |
 
## Dataset
 
This project uses the **Amazon Fine Food Reviews** dataset (568,454 reviews).
 
**Download:** https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
 
Place the `Reviews.csv` file in the same folder as the notebook before running it. The file is not included in this repository due to its size (~300MB).
 
## Requirements
 
```
pip install spacy nltk pandas matplotlib seaborn scikit-learn wordcloud sentence-transformers torch
python -m spacy download en_core_web_sm
```
 
## Project Structure
 
```
text-mining-project/
├── README.md
├── sentiment_analysis_project.ipynb   # Main notebook with full pipeline
└── Text_Mining_Paula.pdf              # Project report 
```
 
## Methodology
 
1. **Exploratory Analysis**: Word clouds, POS-tagging, Named Entity Recognition, n-gram extraction, WordNet synonym/antonym analysis
2. **Preprocessing**: Lowercasing, punctuation removal, stopword removal, lemmatisation (SpaCy)
3. **Feature Representations**: TF unigrams, TF bigrams, TF-IDF unigrams, TF-IDF bigrams, TF-IDF trigrams, BERT sentence embeddings
4. **Classification**: Naive Bayes, Logistic Regression, Linear SVM — evaluated with 5-fold stratified cross-validation
5. **Analysis**: Ablation study on preprocessing, error analysis with shallow parsing
## How to Run
 
1. Download the dataset from Kaggle and place `Reviews.csv` in the project folder
2. Install the required libraries (see Requirements above)
3. Open `sentiment_analysis_project_v3.ipynb` in Jupyter Notebook or VS Code
4. Run all cells sequentially (total runtime: ~30-40 minutes)
## Author
 
Paula García — University of Alicante
