import json
import logging
import os
import sys
import time

import pyzmail

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")


class Importer(object):
    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

    def __init__(self, arg_document_count_limit=sys.maxint, arg_process_text_part=True, arg_process_html_part=False,
                 arg_process_both_empty=False):
        self.document_count_limit = arg_document_count_limit
        self.process_text_part = arg_process_text_part
        self.process_html_part = arg_process_html_part
        self.process_both_empty = arg_process_both_empty

    def process_folder(self, arg_folder):
        document_count = 0
        no_references_count = 0
        references_count = 0
        message_id_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                if document_count < self.document_count_limit:
                    current_full_file_name = os.path.join(root, current)
                    if document_count % 1000 == 0 and document_count > 0:
                        logging.debug("%d %s", document_count, current_full_file_name)
                    references = self.get_references(current_full_file_name)
                    if references.has_key('references'):
                        references_count += 1
                    else:
                        no_references_count += 1
                    document_count += 1
                    if references.has_key('message-id'):
                        message_id_count += 1
        logging.info('documents : %d message-id: %d references: %d no references: %d' % (
        document_count, message_id_count, references_count, no_references_count))

    target_encoding = 'utf-8'

    # https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
    clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

    def clean_address(self, argvalue):
        result = str(argvalue)
        for token in self.clean_address_tokens:
            if token in result:
                result = result.replace(token, ' ')
        return result.lower().strip()

    def get_references(self, current_file):
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
            if 'References' in message.keys():
                references = message['References'].split(' ')
                result['references'] = references
        return result


def run():
    start_time = time.time()

    with open('real-settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        input_folder = data['input_folder']
        document_count_limit = data['document_count_limit']
        if document_count_limit == -1:
            document_count_limit = sys.maxint
        process_text_part = data['process_text_part']
        process_html_part = data['process_html_part']
        process_both_empty = data['process_both_empty']

    instance = Importer(arg_document_count_limit=document_count_limit, arg_process_text_part=process_text_part,
                        arg_process_html_part=process_html_part, arg_process_both_empty=process_both_empty)
    instance.process_folder(input_folder)

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
