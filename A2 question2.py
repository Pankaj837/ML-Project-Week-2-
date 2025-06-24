import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    contractions = {"don't": "do not", "can't": "cannot", "i'm": "i am"}
    for c, e in contractions.items():
        text = text.replace(c, e)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def avg_word2vec(tokens, model, dim=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# Load Twitter dataset (columns: airline_sentiment, text)
df_twitter = pd.read_csv('tweets.csv')
df_twitter['tokens'] = df_twitter['text'].apply(preprocess_tweet)

# Load Google News Word2Vec model (must be present)
w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

df_twitter['vector'] = df_twitter['tokens'].apply(lambda x: avg_word2vec(x, w2v_model))
X_twitter = np.vstack(df_twitter['vector'])
y_twitter = df_twitter['airline_sentiment']

X_train_twit, X_test_twit, y_train_twit, y_test_twit = train_test_split(X_twitter, y_twitter, test_size=0.2, random_state=42)
clf_twitter = LogisticRegression(multi_class='multinomial', max_iter=1000)
clf_twitter.fit(X_train_twit, y_train_twit)
y_pred_twit = clf_twitter.predict(X_test_twit)
print(f"Twitter Sentiment Test Accuracy: {accuracy_score(y_test_twit, y_pred_twit):.4f}")

def predict_tweet_sentiment(model, w2v_model, tweet):
    tokens = preprocess_tweet(tweet)
    vector = avg_word2vec(tokens, w2v_model).reshape(1, -1)
    return model.predict(vector)[0]
