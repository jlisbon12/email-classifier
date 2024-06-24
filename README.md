# Email Classifier

Email remains a vital communication tool in professional settings. However, the prevalence of unsolicited bulk emails, commonly known as spam, can significantly hinder productivity by cluttering inboxes. This project develops a sophisticated classifier designed to identify and filter out spam emails efficiently.


## Problem Statement and Key Features

Filtering spam is a pivotal machine learning application, with challenges stemming from the high-dimensional feature space of text data and the sheer volume of documents. This project leverages a robust voting ensemble classifier and implements topic modeling to effectively categorize emails.

Key Features:

- Data Preprocessing: Includes tokenization and stopword removal to clean and prepare the data.
- Hyperparameter Tuning: Utilizes RandomizedSearchCV for optimizing model parameters.
- Ensemble Learning: Employs a VotingClassifier that integrates Naive Bayes, SVM, and Neural Network models to enhance prediction accuracy.
- Topic Modeling: Applies Latent Dirichlet Allocation (LDA) for categorizing non-spam emails, providing insights into common themes.
- Evaluation Metrics: Uses a confusion matrix and classification reports to measure model performance comprehensively.

## Usage Instructions

To use this model, follow these simple steps:

1. Download the Jupyter notebook and the encoded email dataset.
2. Ensure essential libraries such as NumPy and SciKit-Learn are installed.
3. Execute the notebook cells sequentially to preprocess data, train the model, and evaluate its performance.

## Model Performance

The classifier currently achieves an impressive 97% accuracy on the test dataset. Further enhancements could be realized by refining the hyperparameters and adjusting the number of topics used in LDA.