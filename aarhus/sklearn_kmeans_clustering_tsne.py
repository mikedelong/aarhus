import json
import logging
import os
import re
import string
import time

import matplotlib.pyplot as pyplot
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

from aarhus.aarhus import custom_stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
              if word.islower()]
    return "".join(
        [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip()


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    # todo remove touchup list ?
    if True:
        result = [stemmer.stem(t) for t in filtered_tokens]
    else:
        stems = [stemmer.stem(t) for t in filtered_tokens]
        result = [stem for stem in stems if stem not in touchup_list]
    return result


def clean_address(arg_value):
    result = str(arg_value)
    for token in clean_address_tokens:
        if token in result:
            result = result.replace(token, ' ')
    return result.lower().strip()


start_time = time.time()

stemmer = SnowballStemmer("english")

touchup_list = custom_stopwords.get_specific_stopwords()

# https://groups.google.com/forum/#!topic/microsoft.public.outlookexpress.general/oig7-xNFISg
clean_address_tokens = ['=?us-ascii?Q?', '=0D=0A_=28', '=?utf-8?Q?', '=29?=', '=0D=0A']

with open('./sklearn_kmeans_clustering.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    # model_file_name = data['model_file_name']
    cluster_count = data['cluster_count']
    random_state = data['random_state']
    max_df = data['max_df']
    min_df = data['min_df']

target_encoding = 'utf-8'

stopwords = custom_stopwords.get_stopwords()

file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))
# truncate
if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))

documents = [open(file_name, 'r').read() for file_name in file_names]

tfidf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=200000, min_df=min_df, stop_words='english',
                                   use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 2), decode_error='ignore',
                                   strip_accents='ascii')

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

X_reduced = TruncatedSVD(n_components=50, random_state=random_state).fit_transform(tfidf_matrix)

X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(X_reduced)

fig = pyplot.figure(figsize=(10, 10))
ax = pyplot.axes(frameon=False)
pyplot.setp(ax, xticks=(), yticks=())
pyplot.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                       wspace=0.0, hspace=0.0)
pyplot.scatter(X_embedded[:, 0], X_embedded[:, 1], marker="x")

pyplot.show()
model = KMeans(n_clusters=cluster_count, random_state=random_state)

model.fit(tfidf_matrix)
#
# # todo write terms to a file for later viewing
# terms = tfidf_vectorizer.get_feature_names()
# logging.debug('we have %d terms/feature names' % len(terms))
# terms_out_file = 'sklearn_kmeans_terms.csv'
# with open(terms_out_file, 'w') as terms_out_fp:
#     for item in terms:
#         terms_out_fp.write("%s\n" % item)
# logging.debug('wrote terms to %s' % terms_out_file)
#
# clusters = model.labels_.tolist()
#
# logging.debug('files: %d cluster values: %d' % (len(file_names), len(clusters)))
# cluster_counter = Counter()
# for item in clusters:
#     cluster_counter[item] += 1
#
# logging.debug(cluster_counter)
# logging.debug('smallest cluster has %d items; largest cluster has %d items' % (
#     min(cluster_counter.values()), max(cluster_counter.values())))
#
