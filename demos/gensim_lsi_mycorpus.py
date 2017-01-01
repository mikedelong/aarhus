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


def get_stopwords():
    # here we try to de-noise by removing tokens we've seen in previous topics with this corpus that we suspect
    # are email artifacts and do not represent any topic semantics
    # 15(667.195): -0.479*"hrcoffice.com" + -0.213*"john.podesta" + -0.199*"gmmb.com" + 0.196*"health" + 0.186*"group" + -0.185*"bsgco.com" + -0.167*"would" + 0.157*"care" + -0.139*"dschwerin" + -0.138*"aol.com"
    specific_stopwords = ['gmail.com', 'http', 'https', 'mailto', '\'s', 'n\'t', 'hillaryclinton.com',
                          'googlegroups.com', 'law.georgetown.edu', 'javascript', 'wrote', 'email', 'hrcoffice.com',
                          'john.podesta', 'gmmb.com', 'bsgco.com', 'dschwerin', 'aol.com', '//r20.rs6.net/tn.jsp']

    # 0(88184.071): 0.494*"lt" + 0.488*"gt" + 0.377*"span" + 0.373*"/span" + 0.348*"br" + 0.218*"amp" + 0.122*"nbsp" + 0.119*"cite" + 0.112*"blockquot" + 0.111*"/blockquot"
    # 4(1275.315): -0.372*"style=" + -0.266*"class=" + -0.250*"width=" + -0.220*"td" + -0.220*"/td" + -0.220*"tr" + -0.220*"/tr" + -0.158*"color" + -0.153*"said" + -0.143*"/strong"

    html_stopwords = ['lt', 'gt', 'span', '/span', 'br', 'amp', 'nbsp', 'blockquot', 'cite', '/blockquote',
                      'style=', 'class=', 'width=', 'td', '/td', 'tr', '/tr', '/strong']

    common_words_to_ignore = ['say', 'said', 'would']

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
