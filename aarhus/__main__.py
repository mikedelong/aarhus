import dateutil.parser
import hashlib
import json
import logging
import os
import re
import string
import sys
import time

import bs4
import elasticsearch
import elasticsearch.helpers
import nltk
import pyzmail
from nltk.stem.snowball import SnowballStemmer

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
        self.stemmer = SnowballStemmer("english")

        pass

    # http://brandonrose.org/clustering (with some modifications)
    def strip_proppers(self, arg_text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
        tokens = [word for sent in nltk.sent_tokenize(arg_text) for word in nltk.word_tokenize(sent)
                  if word.islower()]
        # todo get the startswiths and endswiths right here
        return "".join(
            [" " + i if not i.startswith("'") and not i.startswith("/") and not i.endswith(
                "=") and i not in string.punctuation else i for i in tokens]).strip()

    # http://brandonrose.org/clustering
    def tokenize_and_stem(self, arg_text):
        # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
        tokens = [current_word for sent in nltk.sent_tokenize(arg_text) for current_word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        stems = [self.stemmer.stem(token) for token in filtered_tokens]
        return stems

    def process_folder(self, arg_folder, arg_bulk_upload, arg_document_type, arg_buffer_limit, arg_server,
                       arg_index_name, arg_kmeans_cluster_dictionary):
        document_count = 0
        document_buffer = []
        indexed_count = 0
        error_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                if document_count < self.document_count_limit:
                    current_full_file_name = os.path.join(root, current)
                    # logging.debug("%d %s", document_count, current_full_file_name)

                    current_json, document_id = self.get_json(current_full_file_name,
                                                              arg_process_text_part=self.process_text_part,
                                                              arg_process_html_part=self.process_html_part,
                                                              arg_process_both_empty=self.process_both_empty,
                                                              arg_kmeans_cluster_dictionary=arg_kmeans_cluster_dictionary)
                    # logging.debug(current_json)
                    document_count += 1
                    try:
                        if arg_bulk_upload:
                            wrapped = {'_type': arg_document_type, '_source': current_json}
                            document_buffer.append(wrapped)

                            if len(document_buffer) == arg_buffer_limit:
                                try:
                                    index_result = elasticsearch.helpers.bulk(arg_server, document_buffer,
                                                                              index=arg_index_name,
                                                                              request_timeout=1000)
                                    # logging.debug(index_result)
                                    indexed_count += len(document_buffer)
                                    document_buffer = []
                                except elasticsearch.exceptions.ConnectionTimeout as connectionTimeout:
                                    logging.warn(connectionTimeout)
                                    document_buffer = []
                        else:
                            index_result = arg_server.index(index=arg_index_name, doc_type=arg_document_type,
                                                            body=current_json, id=document_id)
                            indexed_count += 1
                            logging.debug("id: %s, result: %s", document_id, index_result)
                    except elasticsearch.exceptions.SerializationError as serializationError:
                        logging.warn(serializationError)
                        error_count += 1
        # need to flush the pending buffer
        if arg_bulk_upload and len(document_buffer) > 0:
            index_result = elasticsearch.helpers.bulk(arg_server, document_buffer, index=arg_index_name)
            logging.debug(index_result)

    target_encoding = 'utf-8'

    # https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
    clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

    def clean_address(self, argvalue):
        result = str(argvalue)
        for token in self.clean_address_tokens:
            if token in result:
                result = result.replace(token, ' ')
        return result.lower().strip()

    def get_json(self, current_file, arg_process_text_part, arg_process_html_part, arg_process_both_empty,
                 arg_kmeans_cluster_dictionary):
        result = {'original_file': current_file}
        with open(current_file, 'rb') as fp:
            message = pyzmail.message_from_file(fp)
            # todo clean up internal whitespace
            senders = message.get_addresses('from')
            result['sender'] = [item[i] for i in [0, 1] for item in senders]
            result['short_sender'] = [item.split('@')[0] for item in result['sender']]
            clean_senders = [self.clean_address(item[1]) for item in senders]
            result['clean_sender'] = clean_senders

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
                        body = payload.decode(charset, 'ignore').encode(self.target_encoding)
                    except LookupError as lookupError:
                        if text_part.charset == 'iso-8859-8-i':
                            body = payload.decode('iso-8859-8', 'ignore').encode(self.target_encoding)
                        else:
                            body = payload.decode('utf-8', 'ignore').encode(self.target_encoding)
                            logging.warn('lookup error %s', lookupError)
                else:
                    body = payload.decode('utf-8', 'ignore').encode(self.target_encoding)
                result['body'] = body

                short_file_name = os.path.basename(current_file)
                result['kmeans_cluster'] = arg_kmeans_cluster_dictionary[short_file_name]

            elif message.html_part is not None and arg_process_html_part:
                payload = message.html_part.part.get_payload()
                payload_text = bs4.BeautifulSoup(payload, 'lxml').get_text().strip()
                charset = message.html_part.charset if message.html_part.charset is not None else 'utf-8'
                result['body'] = payload_text.decode(charset, 'ignore').encode(self.target_encoding)
            elif arg_process_both_empty:
                logging.warn('both text_part and html_part are None: %s', current_file)
            else:
                logging.warn('not processing %s', current_file)

            if 'body' in result.keys():
                if len(result['body']) == 0:
                    result['empty_body'] = True

            if 'Message-Id' in message.keys():
                result['messaage-id'] = message['Message-Id']
            if 'In-Reply-To' in message.keys():
                result['in-reply-to'] = message['In-Reply-To']
            if 'References' in message.keys():
                result['references'] = message['References'].split(' ')

            md5 = hashlib.md5()
            with open(current_file, 'rb') as fp:
                md5.update(fp.read())

            return result, md5.hexdigest()


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
        elasticsearch_host = data['elasticsearch_host']
        elasticsearch_port = data['elasticsearch_port']
        elasticsearch_index_name = data['elasticsearch_index_name']
        elasticsearch_document_type = data['elasticsearch_document_type']
        elasticsearch_batch_size = data['elasticsearch_batch_size']
        kmeans_cluster_file_name = data['kmeans_cluster_file_name']

    # get the connection to elasticsearch
    elasticsearch_server = elasticsearch.Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])

    if elasticsearch_server.indices.exists(elasticsearch_index_name):
        elasticsearch_server.indices.delete(elasticsearch_index_name)
    elasticsearch_server.indices.create(elasticsearch_index_name)

    kmeans_cluster_dictionary = json.load(open(kmeans_cluster_file_name, 'r'))

    mapping = {
        elasticsearch_document_type: {
            'properties': {
                'subject': {
                    'type': 'string',
                    'fields': {
                        'raw': {
                            'type': 'string',
                            'index': 'not_analyzed'
                        }
                    }
                },
                'sender': {
                    'type': 'string',
                    'fields': {
                        'raw': {
                            'type': 'string',
                            'index': 'not_analyzed'
                        }
                    }
                },
                'sender_clean': {
                    'type': 'string',
                    'fields': {
                        'raw': {
                            'type': 'string',
                            'index': 'not_analyzed'
                        }
                    }
                },
                'party': {
                    'type': 'string',
                    'fields': {
                        'raw': {
                            'type': 'string',
                            'index': 'not_analyzed'
                        }
                    }
                },
                'recipient_clean': {
                    'type': 'string',
                    'fields': {
                        'raw': {
                            'type': 'string',
                            'index': 'not_analyzed'
                        }
                    }
                }
            }
        }
    }
    elasticsearch_server.indices.put_mapping(index=elasticsearch_index_name, doc_type=elasticsearch_document_type,
                                             body=mapping)

    instance = Importer(arg_document_count_limit=document_count_limit, arg_process_text_part=process_text_part,
                        arg_process_html_part=process_html_part, arg_process_both_empty=process_both_empty)
    instance.process_folder(input_folder, True, elasticsearch_document_type, elasticsearch_batch_size,
                            elasticsearch_server, elasticsearch_index_name, kmeans_cluster_dictionary)

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
