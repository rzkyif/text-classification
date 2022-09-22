import re
import nltk
from data import load_with_datasets
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
import gensim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_score, recall_score
from .train import utils_preprocess_text

def start_test():
  data = load_with_datasets()

  text = data['train']['text']
  labels = data['train']['label']

  test_text = data['test']['text']
  test_labels = data['test']['label']
  
  labels = list(map(lambda x: 1 if x == 'yes' else 0, labels))
  test_labels = list(map(lambda x: 1 if x == 'yes' else 0, test_labels))

  tokenized = []
  for i in range(0, len(text)):
    tokenized.append(utils_preprocess_text(text[i]))

  test_tokenized = []
  for i in range(0, len(test_text)):
    test_tokenized.append(utils_preprocess_text(test_text[i]))

  model = gensim.models.Word2Vec.load("w2v.model")

  X_train = tokenized
  y_train = labels
  X_test = test_tokenized
  y_test = test_labels

  words = set(model.wv.index_to_key)
  X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words], dtype=object) for ls in X_train], dtype=object)
  X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words], dtype=object) for ls in X_test], dtype=object)

  X_train_vect_avg = []
  for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(100, dtype=float))
          
  X_test_vect_avg = []
  for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(100, dtype=float))
  
  rf = RandomForestClassifier()
  rf_model = rf.fit(X_train_vect_avg, y_train)

  y_pred = rf_model.predict(X_test_vect_avg)

  precision = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
  print('Precision: {} / Recall: {} / Accuracy: {}'.format(
      round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
  pass