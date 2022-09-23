import re
import nltk
from data import load_with_datasets
# import StemmerFactory class
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
import gensim
from time import time


def start_train():
  start = time()
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

  # corpus = tokenized + labels
  
  labels = list(map(lambda x: 1 if x == 'yes' else 0, labels))
  test_labels = list(map(lambda x: 1 if x == 'yes' else 0, test_labels))

  X_train = tokenized

  # bisa ad parameter vector_size, window
  mc = 2
  vs = 100
  w = 5
  print(f"Current params: min_count = {mc}, vector_size = {vs}, window = {w}")
  model = gensim.models.Word2Vec(X_train, min_count=mc, vector_size=vs, window=w)
  # model.wv.most_similar('coronavirus')
  model.save("w2v.model")
  end= time()
  print(f"Time elapsed training: {end-start}")
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
    # Stemming (remove -ing, -ly, ...)
    # factory = StemmerFactory()
    # stemmer = factory.create_stemmer()
    # # stemming process
    # lst_text = [stemmer.stem(word) for word in lst_text]
    return lst_text