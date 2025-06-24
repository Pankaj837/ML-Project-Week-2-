import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Corpus as provided
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

# --- Manual TF-IDF Implementation ---
def manual_tf_idf(corpus):
    docs = [doc.split() for doc in corpus]
    vocab = sorted(set(word for doc in docs for word in doc))
    N = len(docs)
    # Document Frequency (DF)
    df = {word: sum(1 for doc in docs if word in doc) for word in vocab}
    # TF-IDF Matrix
    tf_idf_matrix = []
    for doc in docs:
        doc_len = len(doc)
        doc_vector = []
        for word in vocab:
            tf = doc.count(word) / doc_len
            idf = math.log(N / df[word]) if df[word] != 0 else 0
            doc_vector.append(tf * idf)
        tf_idf_matrix.append(doc_vector)
    return np.array(tf_idf_matrix), vocab

manual_tfidf, vocab = manual_tf_idf(corpus)

# --- Scikit-learn CountVectorizer ---
cv = CountVectorizer()
bow_matrix = cv.fit_transform(corpus).toarray()
cv_vocab = cv.get_feature_names_out()

# --- Scikit-learn TfidfVectorizer ---
tfidf = TfidfVectorizer()
sklearn_tfidf = tfidf.fit_transform(corpus).toarray()
tfidf_vocab = tfidf.get_feature_names_out()

# --- Output Results ---
print("=== Vocabulary (Manual) ===")
print(vocab)
print("\n=== Manual TF-IDF Matrix ===")
print(np.round(manual_tfidf, 3))

print("\n=== CountVectorizer Vocabulary ===")
print(list(cv_vocab))
print("=== CountVectorizer Matrix ===")
print(bow_matrix)

print("\n=== TfidfVectorizer Vocabulary ===")
print(list(tfidf_vocab))
print("=== TfidfVectorizer Matrix ===")
print(np.round(sklearn_tfidf, 3))
