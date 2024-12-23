# Run the best model and go.

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import sklearn
import kagglehub
import os
import string
import nltk
import time
import regex as re
import joblib

from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Download the dataset
root_path = os.getcwd()
path = kagglehub.dataset_download("kazanova/sentiment140")

csv_path = os.path.join(path, "training.1600000.processed.noemoticon.csv")
df = pd.read_csv(
    csv_path,
    encoding="latin-1",
    header=None,
    names=["target", "ids", "date", "flag", "user", "text"]
)

df_1=df['target'].value_counts()
df_target=pd.DataFrame(df_1)
df_target=df_target.reset_index()
df_target.columns = ['target', 'count']
df_target['target'] = df_target['target'].apply(lambda x: 1 if x == 4 else x)



def calculate_results(y_true, y_pred):
  """
  Calculates model accuracy, precision, recall and f1 score of a binary classification model.

  Args:
      y_true: true labels in the form of a 1D array
      y_pred: predicted labels in the form of a 1D array

  Returns a dictionary of accuracy, precision, recall, f1-score.
  """
  # Calculate model accuracy
  model_accuracy = accuracy_score(y_true, y_pred) * 100
  # Calculate model precision, recall and f1 score using "weighted average
  model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
  model_results = {"accuracy": model_accuracy,
                  "precision": model_precision,
                  "recall": model_recall,
                  "f1": model_f1}
  return model_results

# 1. Delete HTML Tags
def delete_html_tags(text):
  clean_text = re.sub(r'<.*?>', '', text)
  return clean_text

# 2. Remove Punctuatiton
def remove_punctuation(text):
  return text.translate(str.maketrans('', '', string.punctuation))

# 3. Lowercase the text
def to_lowercase(text):
  return text.lower()

# 4. Replace all the user tags to "@mention"
def remove_mention(text):
  mention_regex = r"@\w+"
  return re.sub(mention_regex, "@mention", text)

# 5. Remove duplicated spaces
def remove_duplicate_whitespace(text):
  return " ".join(text.split())

# 6. Delete stopwords
def remove_stopwords(text):
  stop_words = set(stopwords.words('english'))
  words = text.split()
  filtered_words = [word for word in words if word not in stopwords.words('english')]
  return " ".join(filtered_words)

# 7. Delete urls
def delete_url(text):
  clean_text = re.sub(r'http\S+', '', text)
  return clean_text

# Function to replace chat words
def replace_chat_words(text):
    chat_words = {
        "brb": "be right back",
        "btw": "by the way",
        "omg": "oh my goodness",
        "ttyl": "talk to you later",
        "omw": "on my way",
        "smh": "shaking my head",
        "smdh": "shaking my darn head",
        "lol": "laugh out loud",
        "tbd": "to be determined",
        "imho": "in my humble opinion",
        "imo": "in my opinion",
        "hmu": "hit me up",
        "iirc": "if I remember correctly",
        "lmk": "let me know",
        "og": "original gangsters",
        "ftw": "for the win",
        "nvm": "nevermind",
        "ootd": "outfit of the day",
        "ngl": "not gonna lie",
        "rq": "real quick",
        "iykyk": "if you know, you know",
        "ong": "on god/I swear",
        "yaaas": "yeah",
        "brt": "be right there",
        "sm": "so much",
        "ig": "i guess",
        "wya": "where you at",
        "istg": "i swear to god",
        "hbu": "how about you",
        "atm": "at the moment",
        "asap": "as soon as possible",
        "fyi": "for your information"
    }
    for shorten_word, full_word in chat_words.items():
        text = text.replace(shorten_word, full_word)
    return text

def preprocess_text(text):
  text = delete_html_tags(text)
  text = delete_url(text)
  text = remove_mention(text)
  text = remove_punctuation(text)
  text = to_lowercase(text)
  text = replace_chat_words(text)
  text = remove_stopwords(text)
  text = remove_duplicate_whitespace(text)


  return text

# Function for splitting our sentence into chars
def split_chars(text):
  return " ".join(list(text))




df["text"] = df["text"].apply(preprocess_text)
df["target"] = df["target"].replace(4, 1)
df = df.drop(["ids", "flag", "user", "Unnamed: 0"], axis=1)
df = df.dropna() 

train_dataset, val_dataset = train_test_split(df,
                                              train_size=0.8, # Drop some of the dataset since collab don't give that much cache to work
                                              shuffle=True)

train_df = pd.DataFrame(train_dataset)
val_df = pd.DataFrame(val_dataset)

# Let's get our sentences
train_sentences = train_df["text"].to_list()
val_sentences = val_df["text"].to_list()

# Split every text sentence into the char sentences.
train_char_sentences = [split_chars(sentence) for sentence in train_dataset["text"]]
val_char_sentences = [split_chars(sentence) for sentence in val_dataset["text"]]

one_hot_encoder = OneHotEncoder(sparse_output=False)
train_labels_one_hot = one_hot_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_one_hot = one_hot_encoder.fit_transform(val_df["target"].to_numpy().reshape(-1, 1))

label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_df["target"].to_numpy().reshape(-1, 1))
val_labels_encoded = label_encoder.fit_transform(val_df["target"].to_numpy().reshape(-1, 1))

baseline_model = Pipeline([
    ("tfidf", TfidfVectorizer()),
     ("clf", MultinomialNB())
])

baseline_model = baseline_model.fit(train_sentences, train_labels_encoded)

# Save the model 
joblib.dump(baseline_model, os.path.join(root_path, 'baseline_model.pkl'))
print("Model saved successfully.")
