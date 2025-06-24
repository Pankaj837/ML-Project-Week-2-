import gensim.downloader as api
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

def preprocess_sms(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return tokens

def avg_word2vec(tokens, model):
    dim = model.vector_size
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(dim)

# Load SMS Spam dataset (columns: Label, Message)
df_sms = pd.read_csv('spam.csv', encoding='latin-1')
# Check and rename columns if needed
if 'v1' in df_sms.columns and 'v2' in df_sms.columns:
    df_sms = df_sms[['v1', 'v2']]
    df_sms.columns = ['label', 'message']
elif 'Label' in df_sms.columns and 'Message' in df_sms.columns:
    df_sms = df_sms[['Label', 'Message']]
    df_sms.columns = ['label', 'message']
else:
    raise ValueError("CSV must contain columns 'v1' and 'v2' or 'Label' and 'Message'.")

df_sms['tokens'] = df_sms['message'].apply(preprocess_sms)

# Load a small pretrained embedding model (GloVe 100d)
w2v_model = api.load('glove-wiki-gigaword-100')  # Downloads automatically, 100D

df_sms['vector'] = df_sms['tokens'].apply(lambda x: avg_word2vec(x, w2v_model))
X_sms = np.vstack(df_sms['vector'])
y_sms = df_sms['label'].map({'ham': 0, 'spam': 1})

X_train_sms, X_test_sms, y_train_sms, y_test_sms = train_test_split(X_sms, y_sms, test_size=0.2, random_state=42)
clf_sms = LogisticRegression(max_iter=1000)
clf_sms.fit(X_train_sms, y_train_sms)
y_pred_sms = clf_sms.predict(X_test_sms)
print(f"SMS Spam Test Accuracy: {accuracy_score(y_test_sms, y_pred_sms):.4f}")

def predict_message_class(model, w2v_model, message):
    tokens = preprocess_sms(message)
    vector = avg_word2vec(tokens, w2v_model).reshape(1, -1)
    pred = model.predict(vector)
    return 'spam' if pred[0] == 1 else 'ham'
