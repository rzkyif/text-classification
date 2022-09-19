import re
import nltk
from data import load_with_datasets
# import StemmerFactory class
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
import gensim
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score


def start_train():
  data = load_with_datasets()

  text = data['train']['text']
  labels = data['train']['label']

  test_text = data['test']['text']
  test_labels = data['test']['label']

  tokenized = []
  for i in range(0, len(text)):
    tokenized.append(utils_preprocess_text(text[i]))
  # all_words = sum(tokenized, [])

  test_tokenized = []
  for i in range(0, len(test_text)):
    test_tokenized.append(utils_preprocess_text(test_text[i]))

  corpus = tokenized + labels
  
  labels = list(map(lambda x: 1 if x == 'yes' else 0, labels))
  test_labels = list(map(lambda x: 1 if x == 'yes' else 0, test_labels))

  X_train = tokenized
  y_train = labels
  X_test = test_tokenized
  y_test = test_labels

  # bisa ad parameter vector_size, window
  model = gensim.models.Word2Vec(X_train, min_count=1)
  model.wv.most_similar('coronavirus')

  words = set(model.wv.index_to_key)
  X_train_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in X_train])
  X_test_vect = np.array([np.array([model.wv[i] for i in ls if i in words]) for ls in X_test])

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
  
  # print(X_test_vect)
  pass

def utils_preprocess_text(text):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    lst_stopwords = nltk.corpus.stopwords.words("indonesian")
    lst_stopwords += ["nya", "yg", "ya", "aja", "kalo", "udah", "ku", "gt", "jd", "sih", "gw", "tp", "sdh", "krn", "jg"]
    text = re.sub(r"\d+", "", text)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    # ## Stemming (remove -ing, -ly, ...)
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    # # stemming process
    # lst_text = [stemmer.stem(word) for word in lst_text]
    
    # ps = nltk.stem.porter.PorterStemmer()
    # lst_text = [ps.stem(word) for word in lst_text]
                
    # ## Lemmatisation (convert the word into root word)
    
    # lem = nltk.stem.wordnet.WordNetLemmatizer()
    # lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    return lst_text