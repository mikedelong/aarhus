import bs4
import json
import logging
import os
import pyzmail
import sys
import time

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")



class Importer(object):
    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

    def __init__(self):
        pass

    def process_folder(self, arg_folder):
        document_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                current_full_file_name = os.path.join(root, current)
                logging.debug("%d %s", document_count, current_full_file_name)
                current_json = self.get_json(current_full_file_name)


    target_encoding = 'utf-8'
    def get_json(self, current_file):
        result = {'original_file': current_file}
        with open(current_file, 'rb') as fp:
            message = pyzmail.message_from_file(fp)
            senders = message.get_addresses('from')
            result['sender'] = senders
            subject = message.get('subject')
            result['subject'] = '' if subject is None else subject.decode('iso-8859-1').encode(self.target_encoding)

            text_part = message.text_part
            if text_part is not None:
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
            elif message.html_part is not None:
                payload = message.html_part.part.get_payload()
                payload_text = bs4.BeautifulSoup(payload, 'lxml').get_text().strip()
                charset = message.html_part.charset if message.html_part.charset is not None else 'utf-8'
                result['body'] = payload_text.decode(charset, 'ignore').encode(self.target_encoding)
            else:
                logging.warn('both text_part and html_part are None: %s', current_file)

            return result

def run():
    start_time = time.time()

    with open('settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        input_folder = data['input_folder']

    instance = Importer()
    instance.process_folder(input_folder)

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
