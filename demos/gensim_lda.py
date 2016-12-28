import json
import logging
import numpy
import os
import re
import string

import nltk
from gensim import corpora, models
from nltk.stem.snowball import SnowballStemmer


logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer("english")


def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
              if word.islower()]
    return "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


with open('./gensim_lda_input.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    topics_count = data['topics_count']
    top_words_count = data['top_words_count']
    lda_passes = data['passes']
    model_file_name = data['model_file_name']

file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))
# truncate
if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))

# documents = []
# for file_name in file_names:
#     with open(file_name, 'r') as file_pointer:
#         document = file_pointer.read().decode('utf-8', 'ignore').encode('ascii', 'ignore')
#         documents.append(document)

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

model = models.LdaModel(corpus, num_topics=topics_count, id2word=dictionary, update_every=5, chunksize=10000,
                        passes=lda_passes, distributed=False, random_state=0)

logging.debug('built LDA model')

model.save(model_file_name)
logging.debug('saved LDA model as %s' % model_file_name)
model.show_topics(num_topics=topics_count)

topics_matrix = model.show_topics(formatted=False, num_words=top_words_count)
logging.debug(topics_matrix)
# topics_matrix = numpy.array(topics_matrix)
#
# topic_words = topics_matrix[:, :, 1]
# for i in topic_words:
#     logging.debug(([str(word) for word in i]))
#     logging.debug()

# now let's go back and check some documents and see what their topics are

model.print_topics(-1)

