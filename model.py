import re
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pickle
import joblib

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from nltk.corpus import stopwords
from sklearn.ensemble import VotingClassifier
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import sagemaker
english_stopwords = stopwords.words('english')
nltk.download('wordnet')
nltk.download('stopwords')



def find_html_tags(message):
    return int("<" and ">" in message)

def find_from(message):
    return int("from" in message)

def link_present(message):
    return int("http://" in message)

def num_words(message):
    return len(message)

def signature(message):
    return int("thank" in message)

def keywords_presence(message):
    keywords = ['free', 'offer', 'limited', 'time', 'only', 'discount', 'click', 'buy', 'now', 'order', 'subscribe', 'trial', 'bonus', 'dear', 'friend', 'amazing', 'deal', 'winner', 'lowest', 'price', 'risk', 'money', 'back', 'guarantee', 'congratulations', 'prize', 'winner', 'congrats', 'spam', 'spamming']
    for word in keywords:
        if word in message:
            return int(True)
    return int(False)
        
def num_nums(message):
    return sum(c.isdigit() for c in message)

def num_phone(message):
    if re.search(r'1\s?-\s?888\s?-\s?\d{3}\s?-\s?\d{4}',str(message)):
        return True
    else:
        return False

def clean_text(message,english_stopwords):
    message = message.lower()
    message = re.sub(r'[^\w\s]', '', message)
    message = re.sub(r'[^a-z0-9\s]','',message)
    message = re.sub(r'\d+', '', message)
    message = message.strip()
    message = word_tokenize(message)
    message = [word for word in message if word not in english_stopwords]
    message = [WordNetLemmatizer().lemmatize(word) for word in message]
    message = ' '.join(message)
    return message


def preprocess(df):
    df['Body'] = df['Body'].astype(str)
    df['html_tags'] = df['Body'].apply(find_html_tags)
    df['from'] = df['Body'].apply(find_from)
    df['link'] = df['Body'].apply(link_present)
    df['num_words'] = df['Body'].apply(num_words)
    df['signature'] = df['Body'].apply(signature)
    df['keywords'] = df['Body'].apply(keywords_presence)
    df['num_nums'] = df['Body'].apply(num_nums)
    df['clean_text'] = df['Body'].apply(lambda x:clean_text(english_stopwords=english_stopwords,message=x))
    return df

def check(message,given_df=False):
    df = pd.DataFrame([message], columns=['Body'])
    df = preprocess(df)
    print('preprocessing done')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectoriser = pickle.load(f)
    # model = joblib.load('voting_classifier.joblib')
    with open('iris_classifier_model.pkl', 'rb') as f:  
        model = pickle.load(f)
    transformed_df = vectoriser.transform(df['clean_text']).toarray()
    print('vectorising done')
    complete_df = np.concatenate((transformed_df,df[['html_tags','from','link','num_words','signature','keywords','num_nums']]),axis=1)
    print('about to start prediction')
    if given_df:
        return complete_df
    else:
        return model.predict(complete_df)
    
if __name__=='__main__':
    print(check("Hello this is spam"))