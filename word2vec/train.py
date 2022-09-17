import re
import nltk
from data import load_with_datasets
# import StemmerFactory class
# from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
# create stemmer
import gensim
from gensim import corpora, similarities
from gensim.models import Word2Vec, KeyedVectors


def start_train():
  data = load_with_datasets()
  text = data['train']['text']
  labels = data['train']['label']

  
  
  # updated = train.map(utils_preprocess_text)
  tokenized = []
  for i in range(0, len(text)):
    # text[i] = utils_preprocess_text(text[i])
    tokenized.append(utils_preprocess_text(text[i]))
  all_words = sum(tokenized, [])

  corpus = tokenized + labels
  model = gensim.models.Word2Vec(tokenized, min_count=1)
  model.wv.most_similar('coronavirus')

  # count = nltk.FreqDist(all_words)

    
  print(model)
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