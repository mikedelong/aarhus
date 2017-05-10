import logging
import collections
import sys
import textract
import json

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)
stop_words = ['of', 'and', 'in', 'the', 'which', 'as', 'to', 'by', 'be', 'is', 'was', 'were', 'on', 'new', 'it', 'had',
              'for', 'would', 'from', 'an', 'been', 'not', 'this']
with open('frequencies-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_file = data['input_file']

text = textract.process(input_file)

words = [word for word in text.lower().split() if len(word) > 1 and word not in stop_words]

logging.debug(len(words))

counts = collections.Counter(words)

logging.debug(counts)

logging.debug(len(counts))