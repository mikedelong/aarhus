import collections
import glob
import json
import logging
import os.path
import pickle
import sys

import nltk.corpus
import textract
from nltk.stem.snowball import SnowballStemmer

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

logging.debug('started.')

input_file = None
input_folder = None
with open('frequencies-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    if 'input_file' in data.keys():
        input_file = data['input_file']
    elif 'input_folder' in data.keys():
        input_folder = data['input_folder']

# most_count = 20
limit = sys.maxint
limit = 1

file_names = list()
words = []
# todo fill this in after doing the folder case
if input_file is not None:
    short_name = os.path.basename(input_file)
    file_names.append(short_name)
    text = textract.process(input_file)
    current_words = [word.rstrip('?:!.,;') for word in text.split()]

current_most = None
file_count = 0
per_file_most = list()
if input_folder is not None:
    if not input_folder.endswith('/'):
        input_folder += '/'
    pathname = input_folder + '*.pdf'
    for this_file in glob.glob(pathname=pathname):

        if file_count < limit:
            short_name = os.path.basename(this_file)
            file_names.append(short_name)
            logging.debug('%d : %s' % (file_count, this_file))
            text = textract.process(this_file)
            current_words = [word for word in text.split()]
            current_words = [word.rstrip('?:!.,;') for word in current_words]
            for index, word in enumerate(current_words):
                if index > 0:
                    w0 = current_words[index-1]
                    if len(word) > 0 and len(w0) > 0 and  word[0].isupper()  and w0[0].isupper():
                        logging.debug('%s %s' % (current_words[index-1], word))
            file_count += 1

logging.debug('done.')