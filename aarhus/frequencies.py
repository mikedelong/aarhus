import logging
import collections
import sys
import textract
import json
import glob
import nltk.corpus

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)
stop_words = set(nltk.corpus.stopwords.words('english'))
stop_words.update(['new', 'one', 'may', 'made', 'however', 'would', 'toward', '--', 'two', 'even', 'november',
                   'december', 'march', 'much'])
# todo add punctuation cleanup
input_file = None
input_folder = None
with open('frequencies-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    if 'input_file' in data.keys():
        input_file = data['input_file']
    elif 'input_folder' in data.keys():
        input_folder = data['input_folder']

most_count = 30
limit = sys.maxint
# limit = 1

words = []
if input_file is not None:
    text = textract.process(input_file)
    current_words = [word for word in text.lower().split()]
    current_words = [word.rstrip('?:!.,;') for word in current_words]
    current_words = [word for word in current_words if len(word) > 1 and word not in stop_words]
    words.extend(current_words)

file_count = 0
if input_folder is not None:
    if not input_folder.endswith('/'):
        input_folder += '/'
    pathname = input_folder + '*.pdf'
    for this_file in glob.glob(pathname=pathname):
        if file_count < limit:
            logging.debug(this_file)
            text = textract.process(this_file)
            current_words = [word for word in text.lower().split()]
            current_words = [word.rstrip('?:!.,;') for word in current_words]
            current_words = [word for word in current_words if len(word) > 1 and word not in stop_words]
            words.extend(current_words)
            file_count += 1

logging.debug('%d total words/tokens' % len(words))

counts = collections.Counter(words)
most = counts.most_common(most_count)
logging.debug(counts)
for item in most:
    logging.debug('%s : %d' % item)
logging.debug('%d unique words/tokens' % len(counts))
