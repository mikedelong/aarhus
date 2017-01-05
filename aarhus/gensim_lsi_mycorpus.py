import json
import logging
import os
import re
import string

import nltk
from gensim import corpora, models
from nltk.stem.snowball import SnowballStemmer

import custom_stopwords

import pyzmail

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer("english")

# https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

def clean_address(arg_value):
    result = str(arg_value)
    for token in clean_address_tokens:
        if token in result:
            result = result.replace(token, ' ')
    return result.lower().strip()


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


class MyCorpus(corpora.TextCorpus):
    stopwords = custom_stopwords.get_stopwords()

    def get_texts(self):
        for filename in self.input:  # for each relevant file
            t0 = open(filename).read()
            t1 = t0.decode('utf-8', 'ignore').encode('ascii', 'ignore')
            t2 = strip_proppers(t1)
            t3 = tokenize_and_stem(t2)
            result = [word for word in t3 if word not in self.stopwords]
            yield result

target_encoding = 'utf-8'
def filter_file_by_content(arg_file_name, arg_senders):
    with open(arg_file_name, 'rb') as fp:
        message = pyzmail.message_from_file(fp)
        # todo clean up internal whitespace
        senders = message.get_addresses('from')
        clean_senders = [clean_address(item[1]) for item in senders]

        if len(set(clean_senders) & arg_senders) != 0:
            return None

        text_part = message.text_part
        if text_part is not None:
            charset = text_part.charset
            payload = text_part.get_payload()
            if charset is not None:
                try:
                    body = payload.decode(charset, 'ignore').encode(target_encoding)
                except LookupError as lookupError:
                    if text_part.charset == 'iso-8859-8-i':
                        body = payload.decode('iso-8859-8', 'ignore').encode(target_encoding)
                    else:
                        body = payload.decode('utf-8', 'ignore').encode(target_encoding)
                        logging.warn('lookup error %s', lookupError)
            else:
                body = payload.decode('utf-8', 'ignore').encode(target_encoding)

            body_ascii = body.decode('utf-8', 'ignore').encode('ascii', 'ignore')

            return body_ascii
        else:
            return None

stopwords = custom_stopwords.get_stopwords()
unanalyzed_senders = custom_stopwords.get_unanalyzed_senders()
unanalyzed_senders = set(unanalyzed_senders)

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

file_names = [file_name for file_name in file_names if filter_file_by_content(file_name, unanalyzed_senders) is not None]
logging.debug('after filtering we are using %d files', len(file_names))

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
