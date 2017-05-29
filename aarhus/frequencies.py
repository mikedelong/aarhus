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
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.update(['new', 'one', 'may', 'made', 'however', 'would', 'toward', '--', 'two', 'even', 'november',
                   'december', 'march', 'much', 'many', 'september', 'october', 'among', 'august', 'make', 'although',
                   'view', 'june', 'might', 'went', 'could', 'must', 'way', 'began', 'us', 'also', 'might', 'become',
                   'seems', 'known', 'months', 'end', 'upon', 'need', 'good', 'seemed', 'begin', 'less', 'more',
                   'thus', 'case', 'mean', 'means', 'main', 'february', 'work', 'play', 'form', 'day', 'first',
                   'second', 'hand', 'come', 'become', 'came', 'became', 'views', 'open', 'close', 'closed',
                   'three', 'third', 'second', 'whether', 'take', 'used', 'move', 'almost', 'january',
                   'follow', 'go', 'work', 'u.'])

stemmer = SnowballStemmer('english')

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
# limit = 1

file_names = list()
words = []
if input_file is not None:
    short_name = os.path.basename(input_file)
    file_names.append(short_name)
    text = textract.process(input_file)
    current_words = [word.rstrip('?:!.,;') for word in text.lower().split()]
    current_words = [word for word in current_words if len(word) > 1 and word not in stop_words]
    logging.debug('before stemming we have %d words' % len(current_words))
    current_words = [stemmer.stem(word) for word in current_words]
    logging.debug('after stemming we have %d words' % len(current_words))
    words.extend(current_words)

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
            current_words = [word for word in text.lower().split()]
            current_words = [word.rstrip('?:!.,;') for word in current_words]
            current_words = [word for word in current_words if len(word) > 1 and word not in stop_words]
            logging.debug('before stemming we have %d words' % len(collections.Counter(current_words)))
            current_words = [stemmer.stem(word) for word in current_words]
            logging.debug('after stemming we have %d words' % len(collections.Counter(current_words)))
            current_counts = collections.Counter(current_words)
            # current_most = current_counts.most_common(most_count)
            current_most = current_counts.most_common()
            per_file_most.append(current_most)
            logging.debug(current_most)
            words.extend(current_words)
            file_count += 1

logging.debug('%d total words/tokens' % len(words))

counts = collections.Counter(words)
# most = counts.most_common(most_count)
most = counts.most_common()
logging.debug(counts)

result = {
    'most_common_from_documents': per_file_most,
    'most_common_from_corpus': most,
    'file_names': file_names
}

output_file = './most_common.pickle'
with open(output_file, 'wb') as output_fp:
    pickle.dump(result, output_fp)

logging.debug('wrote most common words data to output file %s.' % output_file)
