import json
import logging
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
    return "".join(
        [" " + i if not i.startswith("'") and not i.startswith("/") and not i.endswith(
            "=") and i not in string.punctuation else i for i in tokens]).strip()


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


def get_stopwords():
    # here we try to de-noise by removing tokens we've seen in previous topics with this corpus that we suspect
    # are email artifacts and do not represent any topic semantics
    specific_stopwords = ['gmail.com', 'http', 'https', 'mailto', '3cmailto', '\'s', 'n\'t', 'hillaryclinton.com',
                          'googlegroups.com', 'law.georgetown.edu', 'javascript', 'wrote', 'email', 'hrcoffice.com',
                          'john.podesta', 'gmmb.com', 'bsgco.com', 'dschwerin', 'aol.com']

    html_stopwords = ['lt', 'gt', 'span', 'br', 'amp', 'nbsp', 'blockquot', 'cite', 'td', 'tr', 'strong/strong', 'tabl',
                      'tbodi', 'lt/span', 'rgba', 'lt/blockquot', 'background-color', 'lt/div', 'lt/span', 'span/span',
                      'br/blockquot', 'media__imag', 'blockquotetype=', 'nbsp/span', 'gt/span', 'rgba/span', 'lt/p',
                      '0in', 'div', 'p', 'n', 'e', '0pt', 'margin-bottom', '-webkit-composition-fill-color', '2f', '3a',
                      'redirect=http', '2fgmf-pillar', 'media__imagesrc=', 'imgalt=', '3e', 'font-weight', 'font-vari',
                      'font-style', 'font-size:14.666666666666666px', 'white-spac']

    common_words_to_ignore = ['say', 'said', 'would', 'go', 'also']

    stopwords = nltk.corpus.stopwords.words('english') + specific_stopwords + html_stopwords + common_words_to_ignore
    return stopwords


class MyCorpus(corpora.TextCorpus):
    stopwords = get_stopwords()

    def get_texts(self):
        for filename in self.input:  # for each relevant file
            t0 = open(filename).read()
            t1 = t0.decode('utf-8', 'ignore').encode('ascii', 'ignore')
            t2 = strip_proppers(t1)
            t3 = tokenize_and_stem(t2)
            result = [word for word in t3 if word not in self.stopwords]
            yield result


with open('./gensim_lsi_mycorpus.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    corpus_file_name = data['corpus_file_name']
    dictionary_file_name = data['dictionary_file_name']
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    model_file_name = data['model_file_name']
    topics_count = data['topics_count']
    top_words_count = data['top_words_count']

file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))
# truncate
if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))

corpus = MyCorpus([file_name for file_name in file_names])

corpus.dictionary.save(dictionary_file_name)

corpora.MmCorpus.save_corpus(corpus_file_name, corpus)

model = models.LsiModel(corpus, num_topics=topics_count, id2word=corpus.dictionary, chunksize=20000, distributed=False,
                        onepass=True)

logging.debug('built LSI model')

model.save(model_file_name)
logging.debug('saved LSI model as %s' % model_file_name)
model.show_topics(num_topics=topics_count)

topics_matrix = model.show_topics(formatted=False, num_words=top_words_count)
logging.debug(topics_matrix)

model.print_topics(-1)
