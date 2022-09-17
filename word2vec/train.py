import re
import nltk
from data import load_with_datasets

def start_train():
  data = load_with_datasets()
  text = data['train']['text']
  labels = data['train']['label']
  
  # updated = train.map(utils_preprocess_text)
  for i in range(0, len(text)):
    text[i] = utils_preprocess_text(text[i])
    
  print(text[5465])
  pass

def utils_preprocess_text(text):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    lst_stopwords = nltk.corpus.stopwords.words("indonesian")
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in 
                    lst_stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    
    ps = nltk.stem.porter.PorterStemmer()
    lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    
    lem = nltk.stem.wordnet.WordNetLemmatizer()
    lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text