import logging
import pickle
import collections
logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

logging.debug('started.')
input_file = './most_common.pickle'
with open(input_file, 'rb') as input_fp:
    data = pickle.load(input_fp)

logging.debug('read preprocessed data from  %s.' % input_file)

for key in data.keys():
    logging.debug('input data has key %s' % key)

file_names = list()
if 'file_names' in data.keys():
    file_names = data['file_names']

counts_from_corpus = collections.Counter()
if 'counts_from_corpus' in data.keys():
    counts_from_corpus = data['counts_from_corpus']

counts_from_documents = list()
if 'counts_from_documents' in data.keys():
    counts_from_documents = data['counts_from_documents']

t0 = counts_from_corpus.most_common(20)

if True:
    for index, item in enumerate(t0):
        logging.debug('%s: %s :: %d' % (index + 1, item[0], item[1]))
    most_words = set([each[0] for each in t0])
    logging.debug(most_words)
    for item in counts_from_documents:
        t1 = item.most_common(10)
        current = set([each[0] for each in t1])
        logging.debug(current.difference(t0))
