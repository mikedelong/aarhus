import json
import logging
import math
import os
import re
import sys
import time

import matplotlib.pyplot as pyplot
import nltk
import pyzmail
import textblob
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import TSNE

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

stemmer = SnowballStemmer("english")


def get_noun_phrases_and_stem(arg_text):
    blob = textblob.TextBlob(arg_text)
    tokens = [word for phrase in blob.noun_phrases for word in nltk.word_tokenize(phrase)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    result = [stemmer.stem(t) for t in filtered_tokens]
    return result


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    result = [stemmer.stem(t) for t in filtered_tokens]
    return result


class Importer(object):
    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

    def __init__(self, arg_document_count_limit=sys.maxint, arg_process_text_part=True, arg_process_html_part=False,
                 arg_process_both_empty=False):
        self.document_count_limit = arg_document_count_limit
        self.process_text_part = arg_process_text_part
        self.process_html_part = arg_process_html_part
        self.process_both_empty = arg_process_both_empty

    def process_folder(self, arg_folder, arg_reference, arg_in_or_out):
        result = []
        document_count = 0
        no_references_count = 0
        references_count = 0
        message_id_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                # first get the references node
                if document_count < self.document_count_limit:
                    current_full_file_name = os.path.join(root, current)
                    if document_count % 1000 == 0 and document_count > 0:
                        logging.debug("%d %s", document_count, current_full_file_name)
                    references = self.get_references(current_full_file_name)
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
                        result.append(current)
                    elif arg_reference not in references.keys() and not arg_in_or_out:
                        result.append(current)

        logging.info('documents : %d message-id: %d references: %d no references: %d' % (
            document_count, message_id_count, references_count, no_references_count))
        return result

    target_encoding = 'utf-8'

    # https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
    clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

    def clean_address(self, argvalue):
        result = str(argvalue)
        for token in self.clean_address_tokens:
            if token in result:
                result = result.replace(token, ' ')
        return result.lower().strip()

    @staticmethod
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
        return result


def convert(arg_input_folder, arg_limit, arg_filter):
    document_count = 0
    result = []
    for root, subdirectories, files in os.walk(arg_input_folder):
        for current in files:
            document_count += 1
            if document_count < arg_limit and current in arg_filter:
                current_full_file_name = os.path.join(root, current)
                result.append(current_full_file_name)
    return result


def run():
    start_time = time.time()

    with open('counts-settings.json') as data_file:
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

    instance = Importer(arg_document_count_limit=document_count_limit, arg_process_text_part=process_text_part,
                        arg_process_html_part=process_html_part, arg_process_both_empty=process_both_empty)

    document_names_of_interest = instance.process_folder(input_folder, reference_of_interest, in_or_out)
    logging.info('found %d documents of interest: %s' % (len(document_names_of_interest), document_names_of_interest))

    # let's take this sample and build a cluster and display it
    file_names = convert(text_input_folder, document_count_limit, document_names_of_interest)

    documents = [open(file_name, 'r').read() for file_name in file_names]
    logging.debug('finished reading documents into the big list')

    # todo make the tokenizer a setting
    tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=max_features, min_df=min_df, stop_words='english',
                                       use_idf=True,
                                       # tokenizer=tokenize_and_stem,
                                       tokenizer=get_noun_phrases_and_stem,
                                       ngram_range=(1, 2),
                                       decode_error='ignore', strip_accents='ascii')
    logging.debug('built the tfidf vectorizer')

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    logging.debug('built the tfidf matrix')

    reduced = TruncatedSVD(n_components=n_components, random_state=random_state).fit_transform(tfidf_matrix)
    logging.debug('built the SVD with %d components', n_components)

    if manifold == 'isomap':
        n_neighbors = int(math.sqrt(len(document_names_of_interest)))
        embedded = Isomap(n_neighbors=n_neighbors, n_components=2, eigen_solver='auto', tol=0, path_method='auto',
                          neighbors_algorithm='auto').fit_transform(reduced)
        logging.debug('built the Isomap model')
    elif manifold == 'tsne' or manifold == 't-sne':
        embedded = TSNE(n_components=2, perplexity=40, verbose=2, random_state=random_state).fit_transform(reduced)
        logging.debug('built the tSNE model')
    elif manifold == 'locallylinear' or manifold == 'locally-linear' or manifold == 'lle':
        n_neighbors = int(math.sqrt(len(document_names_of_interest)))
        embedded = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=2,
                                          reg=0.001, eigen_solver='auto', tol=1e-06,
                                          max_iter=200, method='standard', hessian_tol=0.0001, modified_tol=1e-12,
                                          neighbors_algorithm='auto', random_state=random_state).fit_transform(reduced)
        logging.debug('built the Locally Linear Embedding model')
    elif manifold == 'mlle':
        n_components = 2
        n_neighbors = max(int(math.sqrt(len(document_names_of_interest))), n_components)
        embedded = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                          reg=0.001, eigen_solver='auto', tol=1e-06,
                                          max_iter=200, method='modified', hessian_tol=0.0001, modified_tol=1e-12,
                                          neighbors_algorithm='auto', random_state=random_state).fit_transform(reduced)
        logging.debug('built the Modified Locally Linear Embedding model')
    elif manifold == 'hessian-lle' or manifold == 'hessian':
        n_components = 2
        n_neighbors = max(int(math.sqrt(len(document_names_of_interest))), n_components)
        embedded = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                          reg=0.001, eigen_solver='auto', tol=1e-06,
                                          max_iter=200, method='hessian', hessian_tol=0.0001, modified_tol=1e-12,
                                          neighbors_algorithm='auto', random_state=random_state).fit_transform(reduced)
        logging.debug('built the Hessian model')
    elif manifold == 'locally-tangent' or manifold == 'ltsa':
        n_components = 2
        n_neighbors = max(int(math.sqrt(len(document_names_of_interest))), n_components)
        embedded = LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components,
                                          reg=0.001, eigen_solver='auto', tol=1e-06,
                                          max_iter=200, method='ltsa', hessian_tol=0.0001, modified_tol=1e-12,
                                          neighbors_algorithm='auto', random_state=random_state).fit_transform(reduced)
        logging.debug('built the Locally Tangent Space Alignment model')
    elif manifold == 'multi-dimensional-scaling' or manifold == 'mds':
        embedded = MDS(n_components=2, metric=True, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=6,
                       random_state=random_state, dissimilarity='euclidean').fit_transform(reduced)
        logging.debug('built the Multi-dimensional Scaling model')
    elif manifold == 'non-metric-multi-dimensional-scaling' or manifold == 'nmmds':
        embedded = MDS(n_components=2, metric=False, n_init=4, max_iter=300, verbose=0, eps=0.001, n_jobs=6,
                       random_state=random_state, dissimilarity='euclidean').fit_transform(reduced)
        logging.debug('built the Multi-dimensional Scaling model')

    fig = pyplot.figure(figsize=(12, 12))
    ax = pyplot.axes(frameon=False)
    pyplot.setp(ax, xticks=(), yticks=())
    pyplot.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9, wspace=0.0, hspace=0.0)
    # http://matplotlib.org/api/markers_api.html
    pyplot.scatter(embedded[:, 0], embedded[:, 1], marker=".")  # was x

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))

    pyplot.show()


if __name__ == '__main__':
    run()
