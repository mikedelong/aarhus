
import logging
import pickle

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

logging.debug('started.')
input_file = './most_common.pickle'
data = None
with open(input_file, 'rb') as input_fp:
    data = pickle.load(input_fp)

logging.debug('read preprocessed data from  %s.' % input_file)
