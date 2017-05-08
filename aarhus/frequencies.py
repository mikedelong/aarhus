import logging
import collections
import sys

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

text = 'the quick brown fox jumped over the lazy hound dog'

words = text.split()

counts = collections.Counter(words)

logging.debug(counts)
