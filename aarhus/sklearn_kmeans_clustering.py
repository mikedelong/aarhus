import json
import logging
import os
import re
import string
from collections import Counter
from scipy.cluster.hierarchy import ward, dendrogram

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from aarhus.aarhus import custom_stopwords

import time

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
    stems = [stemmer.stem(t) for t in filtered_tokens]
    # todo remove touchup list ?
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

model = KMeans(n_clusters=cluster_count, random_state=random_state)

model.fit(tfidf_matrix)

# todo write terms to a file for later viewing
terms = tfidf_vectorizer.get_feature_names()
logging.debug('we have %d terms/feature names' % len(terms))
terms_out_file = 'sklearn_kmeans_terms.csv'
with open(terms_out_file, 'w') as terms_out_fp:
    for item in terms:
        terms_out_fp.write("%s\n" % item)
logging.debug('wrote terms to %s' % terms_out_file)

clusters = model.labels_.tolist()

logging.debug('files: %d cluster values: %d' % (len(file_names), len(clusters)))
cluster_counter = Counter()
for item in clusters:
    cluster_counter[item] += 1

logging.debug(cluster_counter)
logging.debug('smallest cluster has %d items; largest cluster has %d items' % (
    min(cluster_counter.values()), max(cluster_counter.values())))

result = {}
for item in zip(file_names, clusters):
    short_file_name = os.path.basename(item[0])
    result[short_file_name] = item[1]
    pass

cluster_out_file = 'sklearn_kmeans_clusters.json'
with open(cluster_out_file, 'w') as fp:
    json.dump(result, fp)
logging.debug('wrote clusters to %s' % cluster_out_file)

if False:
    distances = 1 - cosine_similarity(tfidf_matrix)
    logging.debug('computed distances usng cosine similarity')

    MDS()

    # convert two components as we're plotting points in a two-dimensional plane
    # "precomputed" because we provide a distance matrix
    # we will also specify `random_state` so the plot is reproducible.
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=random_state)

    pos = mds.fit_transform(distances)  # shape (n_components, n_samples)

    linkage_matrix = ward(distances)  # define the linkage_matrix using ward clustering pre-computed distances
    logging.debug('got linkage matrix')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(30, 40))  # set size
    ax = dendrogram(linkage_matrix, orientation="right")  # ), labels=titles);

    plt.tick_params( \
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom='off',  # ticks along the bottom edge are off
        top='off',  # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()  # show plot with tight layout

    # uncomment below to save figure
    plt.savefig('ward_clusters.png', dpi=200)  # save figure as ward_clusters

pass

finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logging.info(
    "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
