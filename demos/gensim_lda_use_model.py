import json
import logging
import os
import random
import re
import string

import nltk
from gensim import corpora
from gensim import utils
from nltk.stem.snowball import SnowballStemmer

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer("english")


def strip_proppers(arg_text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [current_word for sent in nltk.sent_tokenize(arg_text) for current_word in nltk.word_tokenize(sent)
              if current_word.islower()]
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def tokenize_and_stem(arg_text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [current_word for sent in nltk.sent_tokenize(arg_text) for current_word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


with open('./gensim_lda_use_model.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    model_file_name = data['model_file_name']
    random_seed = data['random_seed']

model = utils.SaveLoad.load(model_file_name)
logging.debug('model restored')

file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))

if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))

documents = [open(file_name, 'r').read().decode('utf-8', 'ignore').encode('ascii', 'ignore') for file_name in
             file_names]
logging.debug('documents array has length %d' % len(documents))

preprocess = [strip_proppers(document) for document in documents]
logging.debug('done stripping propers; result has length %d ', len(preprocess))

tokenized_text = [tokenize_and_stem(text) for text in preprocess]
logging.debug('done tokenizing; result has length %d', len(tokenized_text))
specific_stopwords = ['gmail.com', 'http', 'https', 'mailto', '\'s', 'n\'t', 'hillaryclinton.com', 'googlegroups.com',
                      'law.georgetown.edu', 'javascript', 'wrote', 'email']
stopwords = nltk.corpus.stopwords.words('english') + specific_stopwords
logging.debug('imported stopwords; we have %d of them', len(stopwords))
texts = [[word for word in text if word not in stopwords] for text in tokenized_text]
logging.debug('after stopword removal result has length %d', len(texts))

dictionary = corpora.Dictionary(texts)
logging.debug('dictionary has length %d', len(dictionary))

dictionary.filter_extremes(no_below=1, no_above=0.8)
logging.debug('dictionary has length %d', len(dictionary))

corpus = [dictionary.doc2bow(text) for text in texts]
logging.debug('corpus size is %d', len(corpus))

# todo figure out how to map topic numbers back onto topic word collections
# model.print_topics(1)

random.seed(random_seed)
t0 = random.choice(corpus)
t1 = model.get_document_topics(t0)
# todo find a pythonic way to do this
max_value = 0.0
max_key = 0
for item in t1:
    if item[1] > max_value:
        max_value = item[1]
        max_key = item[0]
logging.debug(t1)
logging.debug(max_key)

pass
