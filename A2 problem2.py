import gensim.downloader as api
import pandas as pd
import numpy as np
import re
import nltk
import os  # Added for file check
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|@\w+|#\w+|[^\w\s]', '', text)
    contractions = {"don't": "do not", "can't": "cannot", "i'm": "i am", 
                   "isn't": "is not", "it's": "it is", "i've": "i have"}
    for cont, exp in contractions.items():
        text = text.replace(cont, exp)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def avg_word2vec(tokens, model):
    dim = model.vector_size
    vectors = [model[word] for word in tokens if word in model.key_to_index]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# Verify file existence
if not os.path.isfile('Tweets.csv'):
    raise FileNotFoundError("Tweets.csv not found. Download from: https://raw.githubusercontent.com/satyajeetkrjha/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv")

df_twitter = pd.read_csv('Tweets.csv')

if 'text' not in df_twitter or 'airline_sentiment' not in df_twitter:
    raise ValueError("CSV must contain 'text' and 'airline_sentiment' columns")

# Preprocess
df_twitter['tokens'] = df_twitter['text'].apply(preprocess_tweet)

# Load embedding model
w2v_model = api.load('glove-twitter-25')

# Vectorization
df_twitter['vector'] = df_twitter['tokens'].apply(lambda x: avg_word2vec(x, w2v_model))
X = np.vstack(df_twitter['vector'])
y = df_twitter['airline_sentiment']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
clf = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Twitter Sentiment Test Accuracy: {acc:.4f}")

# Prediction function
def predict_tweet_sentiment(tweet):
    tokens = preprocess_tweet(tweet)
    vector = avg_word2vec(tokens, w2v_model).reshape(1, -1)
    return clf.predict(vector)[0]
