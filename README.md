# Restaurant Review Classification

Binary sentiment classification (positive / negative) of restaurant reviews.
Dataset: [Restaurant Reviews (Kaggle)](https://www.kaggle.com/datasets/d4rklucif3r/restaurant-reviews), 1000 reviews, balanced 500/500.

## Structure

```
data/    Restaurant_Reviews.tsv
impl/    Restaurant_Review_Classification.ipynb   (full pipeline)
result/  model.pickle, test.pickle, vocab.pickle  (generated)
```

Files in `result/` are created by menu option 1:

| File | Contents |
|------|----------|
| `model.pickle` | The trained classifier. Used for predicting new reviews (option 3) and for evaluation (option 2). |
| `test.pickle` | The held-out 30% test set as (features, label) pairs. Used to re-run the evaluation (option 2) without rebuilding. |
| `vocab.pickle` | The list of unigrams/bigrams used as features. Needed to convert a new review into the same features the model was trained on. |

## Approach

- Preprocessing: lowercase, NLTK `word_tokenize`, English stopword removal, and non-alphabetic tokens dropped.
- Negation words (`not`, `no`, `nor`, `don`, `t`, `didn`, `wasn`, `isn`, and the `n't` token) are kept, so "not good" is not reduced to "good".
- Features: unigrams plus bigrams. Each of the N most frequent terms becomes a boolean feature (present / absent in the review).
- The vocabulary is built from the training split only, so no test data leaks into the features.
- Classifier: scikit-learn `LogisticRegression`, wrapped with NLTK's `SklearnClassifier`, with a 70/30 train/test split (seed 42).
- Tuning: `tuneVocab` tries several vocabulary sizes (default 200 to 1500), each averaged over 3 random splits, and returns the best size. The final model is built with that size.
- Evaluation: accuracy, plus precision and recall for the positive class.

## Results

The model reaches roughly 80 to 82% accuracy on held-out reviews, averaged over random splits. A single split is noisy, so the number printed after a build can differ by a few points. The tuning averages are the more reliable figure.

## Setup

Requires Python 3:

```
pip install nltk pandas scikit-learn jupyter
```

The first cell downloads the NLTK data (`punkt`, `punkt_tab`, `stopwords`).

## Run

Open `impl/Restaurant_Review_Classification.ipynb` and run the cells in order:

1. Run the imports/helper cell, then the function cells.
2. Run the tuning cell to find the best vocabulary size (`best`).
3. Run the menu cell and use the interactive menu:

| Option | Action |
|--------|--------|
| 1 | Build the model with the best vocabulary size and print TP/FP/FN, precision, recall, accuracy. Saves the model, test set and vocabulary to `result/`. |
| 2 | Show the saved model's evaluation on the saved test set. |
| 3 | Enter a review and get Positive / Negative. |
| 4 | Exit. |

Use the saved model outside the notebook:

```python
import pickle
from nltk import bigrams
from nltk.tokenize import word_tokenize

with open("result/model.pickle", "rb") as f:
    model = pickle.load(f)
with open("result/vocab.pickle", "rb") as f:
    vocab = pickle.load(f)

tokens = word_tokenize("Great food and friendly staff".lower())
tokens += [" ".join(b) for b in bigrams(tokens)]
features = {v: (v in set(tokens)) for v in vocab}
print(model.classify(features))  # "Positive" or "Negative"
```
