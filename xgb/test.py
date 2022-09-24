import pickle
from data import load_with_datasets
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

def start_test():
  # Preparation
  data = load_with_datasets()
  x_test = data['test']['text']
  y_test = data['test']['label']

  test_size = len(y_test)
  for i in range(test_size):
    if y_test[i] == 'yes':
      y_test[i] = 1
    else:
      y_test[i] = 0

  # Load vector model
  cv = pickle.load(open('count.pkl', 'rb'))
  tfidfv = pickle.load(open('tfidf.pkl', 'rb'))
  x_test_count = cv.transform(x_test)
  x_test_tfidf = tfidfv.transform(x_test)

  # Prediction
  xgb_count_t = XGBClassifier()
  xgb_count_t.load_model('count_tree.model')
  y_pred = xgb_count_t.predict(x_test_count)
  print("Accuracy for Count Vector, Tree model   :",accuracy_score(y_pred, y_test))

  xgb_count_l = XGBClassifier()
  xgb_count_l.load_model('count_linear.model')
  y_pred = xgb_count_l.predict(x_test_count)
  print("Accuracy for Count Vector, Linear model :",accuracy_score(y_pred, y_test))

  xgb_tfidf_t = XGBClassifier()
  xgb_tfidf_t.load_model('tfidf_tree.model')
  y_pred = xgb_tfidf_t.predict(x_test_tfidf)
  print("Accuracy for TFIDF Vector, Tree model   :",accuracy_score(y_pred, y_test))

  xgb_tfidf_l = XGBClassifier()
  xgb_tfidf_l.load_model('tfidf_linear.model')
  y_pred = xgb_tfidf_l.predict(x_test_tfidf)
  print("Accuracy for TFIDF Vector, Linear model :",accuracy_score(y_pred, y_test))
  
  return