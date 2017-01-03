from nltk.corpus import stopwords
import csv
import logging
import os.path

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


html_stopwords_file_name = 'html_stopwords.csv'
if os.path.isfile(html_stopwords_file_name):
    logging.debug('loading html stopwords from %s' % html_stopwords_file_name)
    with open(html_stopwords_file_name, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file)
        html_stopwords = [item for item in csv_reader]
else:
    logging.debug('loading no html stopwords')
    html_stopwords = []

specific_stopwords_file_name = 'specific_stopwords.csv'
if os.path.isfile(specific_stopwords_file_name):
    logging.debug('loading case-specific stopwords from %s' % specific_stopwords_file_name)
    with open(specific_stopwords_file_name, 'rb') as csv_file:
        csv_reader = csv.reader(csv_file)
        specific_stopwords = [item for item in csv_reader]
else:
    logging.debug('loading no case-specific stopwords')
    specific_stopwords = []

ignore_words_file_name = 'ignore_words.csv'
if os.path.isfile(ignore_words_file_name):
    logging.debug('loading common words to ignore from %s' % ignore_words_file_name)
    with open(ignore_words_file_name) as csv_file:
        csv_reader = csv.reader(csv_file)
        common_words_to_ignore = [item for item in csv_reader]
else:
    logging.debug('loading no common words to ignore list')
    common_words_to_ignore = []


def get_stopwords():
    # here we try to de-noise by removing tokens we've seen in previous topics with this corpus that we suspect
    # are email artifacts and do not represent any topic semantics
    result = stopwords.words('english') + specific_stopwords + html_stopwords + common_words_to_ignore
    return result
