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


class MyCorpus(corpora.TextCorpus):
    specific_stopwords = ['gmail.com', 'http', 'https', 'mailto', '\'s', 'n\'t', 'hillaryclinton.com',
                          'googlegroups.com',
                          'law.georgetown.edu', 'javascript', 'wrote', 'email']
    stopwords = nltk.corpus.stopwords.words('english') + specific_stopwords

    def get_texts(self):
        for filename in self.input:  # for each relevant file
            t0 = open(filename).read()
            t1 = t0.decode('utf-8', 'ignore').encode('ascii', 'ignore')
            t2 = strip_proppers(t1)
            t3 = tokenize_and_stem(t2)
            result = [word for word in t3 if word not in self.stopwords]
            yield result


with open('./gensim_lda_input.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    topics_count = data['topics_count']
    top_words_count = data['top_words_count']
    lda_passes = data['passes']
    model_file_name = data['model_file_name']
    dictionary_file_name = data['dictionary_file_name']
    corpus_file_name = data['corpus_file_name']


file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))
# truncate
if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))


corpus = MyCorpus([file_name for file_name in file_names])

corpus.dictionary.save(dictionary_file_name)

corpora.MmCorpus.save_corpus(corpus_file_name, corpus)

model = models.LdaModel(corpus, num_topics=topics_count, id2word=corpus.dictionary, update_every=5, chunksize=10000,
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
