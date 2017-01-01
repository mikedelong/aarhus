import json
import logging
import re
import string

import nltk
from gensim import utils
from gensim.corpora import Dictionary
from nltk.stem.snowball import SnowballStemmer
from gensim.models import LsiModel

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


with open('./gensim_lsi_use_model_mycorpus.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)

    dictionary_file_name = data['dictionary_file_name']
    model_file_name = data['model_file_name']

model = LsiModel.load(model_file_name)
logging.debug('model restored')
logging.debug(model)
model.print_debug()

dictionary = Dictionary.load(dictionary_file_name)
logging.debug(dictionary)

text = u"the quick brown fox jumped over the lazy hound"
document = tokenize_and_stem(strip_proppers(text))
logging.debug(document)

t0 = dictionary.doc2bow(document)
logging.debug(t0)

t1 = model[t0]
logging.debug(t1)

# # todo find a pythonic way to do this
max_value = -1.0
max_key = -1
for item in t1:
    if item[1] > max_value:
        max_value = item[1]
        max_key = item[0]

logging.debug(max_key)
