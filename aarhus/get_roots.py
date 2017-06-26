import json
import logging
import os
import pickle
import sys
import time

import pyzmail

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


def process_folder(arg_folder, arg_reference, arg_in_or_out, arg_document_count_limit):
    result = dict()
    document_count = 0
    no_references_count = 0
    references_count = 0
    message_id_count = 0
    for root, subdirectories, files in os.walk(arg_folder):
        for current in files:
            # first get the references node
            if document_count < arg_document_count_limit:
                current_full_file_name = os.path.join(root, current)
                if document_count % 1000 == 0 and document_count > 0:
                    logging.debug("%d %s", document_count, current_full_file_name)
                references, message = get_references(current_full_file_name)
                if 'references' in references.keys():
                    # if references.has_key('references'):
                    references_count += 1
                else:
                    no_references_count += 1
                document_count += 1
                if 'message-id' in references.keys():
                    # if references.has_key('message-id'):
                    message_id_count += 1

                if arg_reference in references.keys() and arg_in_or_out:
                    # result.append(current)
                    result[current] = message
                elif arg_reference not in references.keys() and not arg_in_or_out:
                    # result.append(current)
                    result[current] = message

    logging.info('documents : %d message-id: %d references: %d no references: %d' % (
        document_count, message_id_count, references_count, no_references_count))
    return result


def get_references(current_file):
    result = {}
    with open(current_file, 'rb') as fp:
        message = pyzmail.message_from_file(fp)
        if 'Message-Id' in message.keys():
            result['message-id'] = message['Message-Id']
        elif 'Message-ID' in message.keys():
            result['message-id'] = message['Message-ID']
        elif 'Message-id' in message.keys():
            result['message-id'] = message['Message-id']
        else:
            logging.warn('no message id in file %s', current_file)
            logging.info([key for key in message.keys()])
        if 'References' in message.keys():
            references = message['References'].split(' ')
            result['references'] = references
        if 'In-Reply-To' in message.keys():
            result['in-reply-to'] = message['In-Reply-To']
    return result, message


# todo get this code to find just the roots of email chains, not the replies
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
        pickle_file = data['pickle_file']

    documents_of_interest = process_folder(input_folder, reference_of_interest, in_or_out, document_count_limit)
    logging.info(
        'found %d documents of interest: %s' % (len(documents_of_interest), sorted(documents_of_interest.keys())))

    with open(pickle_file, 'wb') as output_fp:
        pickle.dump(documents_of_interest, output_fp)
    logging.info('wrote pickled dictionary to %s.' % pickle_file)
    # todo write a reader/processor for the pickle file

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
