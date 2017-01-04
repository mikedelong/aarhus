import json
import logging
import os
import re
import string
from collections import Counter

import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from aarhus.aarhus import custom_stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer("english")


def strip_proppers(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)
              if word.islower()]
    return "".join(
        [" " + i if not i.startswith("'") and not i.startswith("/") and not i.endswith(
            "=") and i not in string.punctuation else i for i in tokens]).strip()


def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it'sown token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


with open('./sklearn_kmeans_clustering.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    # model_file_name = data['model_file_name']
    cluster_count = data['cluster_count']
    random_state = data['random_state']

stopwords = custom_stopwords.get_stopwords()

file_names = [os.path.join(root, current) for root, subdirectories, files in os.walk(input_folder) for current in files]
logging.debug('we have %d files', len(file_names))
# truncate
if max_file_count < len(file_names) and max_file_count != -1:
    file_names = file_names[:max_file_count]
logging.debug('we are using %d files', len(file_names))

documents = [open(file_name, 'r').read().decode('utf-8', 'ignore').encode('ascii', 'ignore') for file_name in
             file_names]

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000, min_df=0.2, stop_words='english', use_idf=True,
                                   tokenizer=tokenize_and_stem, ngram_range=(1, 3))

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

model = KMeans(n_clusters=cluster_count, random_state=random_state)

model.fit(tfidf_matrix)

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

with open('sklearn_kmeans_clusters.json', 'w') as fp:
    json.dump(result, fp)
