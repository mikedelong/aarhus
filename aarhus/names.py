import glob
import json
import logging
import os.path
import re
import string
import sys

import textract

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

logging.debug('started.')

_digits = re.compile('\d')


def contains_digits(arg):
    return bool(_digits.search(arg))


punctuation = set(string.punctuation)
bogeys = set(string.punctuation)
bogeys.remove('-')
input_file = None
input_folder = None
name_token_file = None
bad_token_file = None
limit = 0
with open('frequencies-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    if 'input_file' in data.keys():
        input_file = data['input_file']
    elif 'input_folder' in data.keys():
        input_folder = data['input_folder']
    if 'name_token_file' in data.keys():
        name_token_file = data['name_token_file']
    if 'bad_token_file' in data.keys():
        bad_token_file = data['bad_token_file']
    if 'document_limit' in data.keys():
        limit = int(data['document_limit'])
        if limit == -1:
            limit = sys.maxint

file_names = list()
words = []

name_tokens = set()
if name_token_file is not None:
    with open(name_token_file, 'rb') as tokens_fp:
        content = [each.strip('\n') for each in tokens_fp.readlines()]
        name_tokens = set(content)
bad_tokens = set()
if bad_token_file is not None:
    with open(bad_token_file, 'rb') as tokens_fp:
        content = [each.strip('\n') for each in tokens_fp.readlines()]
        bad_tokens = set(content)

# todo fill this in after doing the folder case
if input_file is not None:
    short_name = os.path.basename(input_file)
    file_names.append(short_name)
    text = textract.process(input_file)
    current_words = [word.rstrip('?:!.,;') for word in text.split()]

current_most = None
file_count = 0
per_file_most = list()
count = 0
zeros_count = 0
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
                    w0 = current_words[index - 1]
                    if len(word) > 0 and len(w0) > 0 and word[0].isupper() and w0[0].isupper():
                        score = 0
                        if w0 in name_tokens:
                            score += 1
                        if w0.isupper() and word.isupper():
                            score -= 1
                        if (w0.isupper() and len(w0)) or (word.isupper() and len(word)) > 1:
                            score -= 1
                        if w0.isdigit() or word.isdigit():
                            score -= 1
                        if w0 in bad_tokens or word in bad_tokens:
                            score -= 1
                        if w0.split('-')[0] in bad_tokens or word.split('-')[0] in bad_tokens:
                            score -= 1
                        if '\'' in word:
                            score -= 1
                        if w0.endswith('\"') or word.endswith('\"'):
                            score -= 1
                        if any([ord(c) > 128 for c in w0 + word]):
                            score -= 1
                        if contains_digits(w0 + word):
                            score -= 1
                        if w0.endswith(')') or word.endswith(')'):
                            count -= 1
                        if w0.endswith('*') or word.endswith('*'):
                            count -= 1
                        if any([each.isdigit() for each in w0 + word]):
                            score -= 1
                        if any([each in bogeys for each in w0 + word]):
                            score -= 1
                        if score >= 0:
                            logging.debug('%d %d [%s] %s %d [%s]' % (score, index, w0,
                                                                     any([each in bogeys for each in w0 + word]),
                                                                     len(w0), word))
                        if score > 0:
                            count += 1
                        elif score == 0:
                            zeros_count += 1
            file_count += 1

logging.debug('total found: %d; zeros found: %d' % (count, zeros_count))
logging.debug('done.')
