
import logging
import pickle
import sys
import time

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

def run():
    start_time = time.time()
    logging.debug('started.')

    # todo move input pickle file name to settings
    input_pickle_file = './tokens.pickle'
    logging.debug('loading tokens dictionary from file %s' % input_pickle_file)
    with open(input_pickle_file, 'rb') as input_fp:
        tokens_dict = pickle.load(input_fp)

    logging.debug(len(tokens_dict))
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
