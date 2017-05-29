import logging
import pickle

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

most = list()
if 'most_common_from_corpus' in data.keys():
    most = data['most_common_from_corpus']

per_file_most = list()
if 'most_common_from_documents' in data.keys():
    per_file_most = data['most_common_from_documents']

if True:
    for index, item in enumerate(most):
        logging.debug('%s: %s :: %d' % (index + 1, item[0], item[1]))
    # logging.debug('%d unique words/tokens' % len(counts))
    most_words = set([each[0] for each in most])
    logging.debug(most_words)
    for item in per_file_most:
        current = set([each[0] for each in item])
        logging.debug(current.difference(most))
