import logging
import sys

import time
import json

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


def run():
    start_time = time.time()

    with open('roots-settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        input_folder = data['input_folder']
        document_count_limit = data['document_count_limit']
        if document_count_limit == -1:
            document_count_limit = sys.maxint
        process_text_part = data['process_text_part']
        process_html_part = data['process_html_part']
        process_both_empty = data['process_both_empty']
        text_input_folder = data['text_input_folder']
        random_state = data['random_state']
        max_df = data['max_df']
        min_df = data['min_df']
        max_features = data['max_features']
        if max_features == -1:
            max_features = None
        n_components = data['n_components']
        reference_of_interest = data['reference']
        # our internal keys are always lowercase, so we want to be sure
        # to use a lowercase reference for comparisons
        reference_of_interest = reference_of_interest.lower()
        in_or_out = data['reference_in']
        in_or_out = bool(in_or_out)
        manifold = data['manifold']
        manifold = str(manifold).lower()

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))

if __name__ == '__main__':
    run()
