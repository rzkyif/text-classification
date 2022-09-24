import re, nltk, pickle
from data import load_with_datasets
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time

def start_train():
  # Preparation
  print("Start training...")
  start = time()

  data = load_with_datasets()
  x_train = data['train']['text']
  y_train = data['train']['label']

  train_size = len(y_train)

  # Preprocessing
  stopwords = nltk.corpus.stopwords.words('indonesian') + ["nya", "yg", "ya", "aja", "kalo", "udah", "ku", "gt", "jd", "sih", "gw", "tp", "sdh", "krn", "jg"]

  for i in range(train_size):
    x_train[i] = re.sub(r"https\st\sco\s.*", "", x_train[i])
    x_train[i] = re.sub(r"\d+", "", x_train[i])
    x_train[i] = re.sub(r"[^\w\s]", '', str(x_train[i]).lower().strip())
    x_train[i] = re.sub(r"[^a-zA-Z\s:]", "", x_train[i])
    x_train[i] = x_train[i].split()
    x_train[i] = [word for word in x_train[i] if word not in stopwords]
    x_train[i] = ' '.join(x_train[i])

  # Vectorization
  cv = CountVectorizer()
  cv.fit(x_train)

  tfidfv = TfidfVectorizer()
  tfidfv.fit(x_train)

  # Save vector model
  pickle.dump(cv, open('count.pkl', 'wb'))
  pickle.dump(tfidfv, open('tfidf.pkl', 'wb'))

  # Training data
  x_train_count = cv.transform(x_train)
  x_train_tfidf = tfidfv.transform(x_train)

  for i in range (train_size):
    if y_train[i] == 'yes':
      y_train[i] = 1
    else:
      y_train[i] = 0
  
  # Training model
  tick = time()
  xgb_count_t = XGBClassifier(booster='gbtree')
  xgb_count_t.fit(x_train_count, y_train)
  xgb_count_t.save_model('count_tree.model')
  print("Count Vector, Tree Model training time   :", time()-tick)

  tick = time()
  xgb_count_l = XGBClassifier(booster='gblinear')
  xgb_count_l.fit(x_train_count, y_train)
  xgb_count_l.save_model('count_linear.model')
  print("Count Vector, Linear Model training time :", time()-tick)
  
  tick = time()
  xgb_tfidf_t = XGBClassifier(booster='gbtree')
  xgb_tfidf_t.fit(x_train_tfidf, y_train)
  xgb_tfidf_t.save_model('tfidf_tree.model')
  print("TFIDF Vector, Tree Model training time   :", time()-tick)

  tick = time()
  xgb_tfidf_l = XGBClassifier(booster='gblinear')
  xgb_tfidf_l.fit(x_train_tfidf, y_train)
  xgb_tfidf_l.save_model('tfidf_linear.model')
  print("TFIDF Vector, Linear Model training time :", time()-tick)
  
  print("Time elapsed:", time()-start)
  print("End training.")
  return