import dateutil.parser
import json
import logging
import os
import sys
import time

import bs4
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
        pass

    def process_folder(self, arg_folder):
        document_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                if document_count < self.document_count_limit:
                    current_full_file_name = os.path.join(root, current)
                    logging.debug("%d %s", document_count, current_full_file_name)
                    current_json = self.get_json(current_full_file_name,
                                                 arg_process_text_part=self.process_text_part,
                                                 arg_process_html_part=self.process_html_part,
                                                 arg_process_both_empty=self.process_both_empty)
                    logging.debug(current_json)
                    document_count += 1

    target_encoding = 'utf-8'

    # https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
    clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

    def clean_address(self, argvalue):
        result = str(argvalue)
        for token in self.clean_address_tokens:
            if token in result:
                result = result.replace(token, ' ')
        return result.lower().strip()

    def get_json(self, current_file, arg_process_text_part, arg_process_html_part, arg_process_both_empty):
        result = {'original_file': current_file}
        with open(current_file, 'rb') as fp:
            message = pyzmail.message_from_file(fp)
            # todo clean up internal whitespace
            senders = message.get_addresses('from')
            result['sender'] = [item[i] for i in [0, 1] for item in senders]
            result['short_sender'] = [item.split('@')[0] for item in result['sender']]
            result['clean_sender'] = [self.clean_address(item[1]) for item in senders]

            # todo clean up internal whitespace
            recipients = message.get_addresses('to') + message.get_addresses('cc') + message.get_addresses('bcc')
            result['recipient'] = recipients
            result['party'] = list(
                ['{name} = {address}'.format(name=item[0], address=item[1]) for item in senders + recipients])
            result['clean_recipient'] = [self.clean_address(item[1]) for item in recipients]
            result['recipient'] = [item[i] for i in [0, 1] for item in recipients]
            result['short_recipient'] = [item.split('@')[0] for item in result['clean_recipient']]

            subject = message.get('subject')
            result['subject'] = '' if subject is None else subject.decode('iso-8859-1').encode(self.target_encoding)

            raw_date = message.get('date')
            if raw_date is not None:
                try:
                    result['date'] = dateutil.parser.parse(raw_date)
                except ValueError as valueError:
                    # todo find a way to deal with these special cases?
                    # we occasionally get a string the parser won't parse e.g. Wed, 17 Dec 2008 12:35:42 -0700 (GMT-07:00)
                    # and we need to drop off the trailing time zone and try to parse again
                    logging.warn('%s %s %s', raw_date, valueError, current_file)
                    pieces = str(raw_date).split('(')
                    result['date'] = dateutil.parser.parse(pieces[0])
            else:
                # todo add special code to handle these?
                logging.warn('no date: %s ', message)

            text_part = message.text_part
            if text_part is not None and arg_process_text_part:
                charset = text_part.charset
                payload = text_part.get_payload()
                if charset is not None:
                    try:
                        result['body'] = payload.decode(charset, 'ignore').encode(self.target_encoding)
                    except LookupError as lookupError:
                        if text_part.charset == 'iso-8859-8-i':
                            result['body'] = payload.decode('iso-8859-8', 'ignore').encode(self.target_encoding)
                        else:
                            result['body'] = payload.decode('utf-8', 'ignore').encode(self.target_encoding)
                            logging.warn('lookup error %s', lookupError)
                else:
                    result['body'] = payload.decode('utf-8', 'ignore').encode(self.target_encoding)
            elif message.html_part is not None and arg_process_html_part:
                payload = message.html_part.part.get_payload()
                payload_text = bs4.BeautifulSoup(payload, 'lxml').get_text().strip()
                charset = message.html_part.charset if message.html_part.charset is not None else 'utf-8'
                result['body'] = payload_text.decode(charset, 'ignore').encode(self.target_encoding)
            elif arg_process_both_empty:
                logging.warn('both text_part and html_part are None: %s', current_file)
            else:
                logging.warn('not processing %s', current_file)

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
