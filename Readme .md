# Assignment 2.1: Text Vectorization Implementation

## Objective
- Manually implement the TF-IDF algorithm.
- Compare with scikit-learn's `CountVectorizer` and `TfidfVectorizer`.
- Explain why common words (like 'the') have different scores.

## Corpus

the sun is a star
the moon is a satellite
the sun and moon are celestial bodies


## Results Example

### Vocabulary

['a', 'and', 'are', 'bodies', 'celestial', 'is', 'moon', 'satellite', 'star', 'sun', 'the']


### Example Output (Document 1)
| Term      | Manual TF-IDF | CountVectorizer | TfidfVectorizer |
|-----------|---------------|-----------------|-----------------|
| the       | 0.000         | 1               | 0.252           |
| sun       | 0.081         | 1               | 0.504           |
| is        | 0.081         | 1               | 0.504           |
| a         | 0.081         | 1               | 0.504           |
| star      | 0.220         | 1               | 0.607           |

### Explanation for Score Differences

- **Manual TF-IDF** uses the formula:  
  `TF = count / doc_length`  
  `IDF = ln(N / df)`  
  For common words like 'the', which appear in every document, IDF becomes `ln(3/3) = 0`, so their TF-IDF is zero.

- **TfidfVectorizer** (scikit-learn) uses **smoothed IDF**:  
  `IDF = ln((1 + N)/(1 + df)) + 1`  
  This prevents zero values, so even common words get a nonzero score.  
  It also applies **L2 normalization** so each document vector has unit length, further affecting the scores.

- **CountVectorizer** simply counts word occurrences, treating all words equally regardless of their importance in the corpus.

### Recommendations for Consistency
- To reproduce manual TF-IDF with scikit-learn, set:
  - `smooth_idf=False`
  - `norm=None`
  - `sublinear_tf=False`

## Conclusion
- Manual TF-IDF and scikit-learn's TfidfVectorizer may yield different results for common words due to smoothing and normalization.
- CountVectorizer does not account for word importance across the corpus.

---

**End of Assignment 2.1**
