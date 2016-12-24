import json
import logging
from os import path
import os
import pyzmail

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('./extract_text.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    if not str(input_folder).endswith('/'):
        input_folder += '/'
    output_folder = data['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not str(output_folder).endswith('/'):
        output_folder += '/'

target_encoding = 'utf-8'
for item in os.listdir(input_folder):
    logging.debug(item)
    if path.isdir(input_folder + item):
        output_subfolder = output_folder + item + '/'
        logging.debug(output_subfolder)
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)
        for current_file in os.listdir(input_folder + item):
            logging.debug(current_file)
            with open(input_folder + item + '/' + current_file, 'rb') as fp:
                message = pyzmail.message_from_file(fp)
                text_part = message.text_part
                if text_part is not None:
                    charset = text_part.charset
                    payload = text_part.get_payload()
                    if charset is not None:
                        try:
                            result= payload.decode(charset, 'ignore').encode(target_encoding)
                        except LookupError as lookupError:
                            if text_part.charset == 'iso-8859-8-i':
                                result= payload.decode('iso-8859-8', 'ignore').encode(target_encoding)
                            else:
                                result = payload.decode('utf-8', 'ignore').encode(target_encoding)
                                logging.warn('lookup error %s', lookupError)
                    else:
                        result = payload.decode('utf-8', 'ignore').encode(target_encoding)
                    with open(output_subfolder + current_file, 'w') as text_file:
                        text_file.write(result)



