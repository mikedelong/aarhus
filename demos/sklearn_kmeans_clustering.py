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


def get_stopwords():
    # here we try to de-noise by removing tokens we've seen in previous topics with this corpus that we suspect
    # are email artifacts and do not represent any topic semantics
    specific_stopwords = ['gmail.com', 'http', 'https', 'mailto', '3cmailto', '\'s', 'n\'t', 'hillaryclinton.com',
                          'googlegroups.com', 'law.georgetown.edu', 'javascript', 'wrote', 'email', 'hrcoffice.com',
                          'john.podesta', 'gmmb.com', 'bsgco.com', 'dschwerin', 'aol.com']

    html_stopwords = ['lt', 'gt', 'span', 'br', 'amp', 'nbsp', 'blockquot', 'cite', 'td', 'tr', 'strong/strong', 'tabl',
                      'tbodi', 'lt/span', 'rgba', 'lt/blockquot', 'background-color', 'lt/div', 'lt/span', 'span/span',
                      'br/blockquot', 'media__imag', 'blockquotetype=', 'nbsp/span', 'gt/span', 'rgba/span', 'lt/p',
                      '0in', 'div', 'p', 'n', 'e', '0pt', 'margin-bottom', '-webkit-composition-fill-color', '2f', '3a',
                      'redirect=http', '2fgmf-pillar', 'media__imagesrc=', 'imgalt=', '3e', 'font-weight', 'font-vari',
                      'font-style', 'font-size:14.666666666666666px', 'white-spac']

    common_words_to_ignore = ['say', 'said', 'would', 'go', 'also']

    stopwords = nltk.corpus.stopwords.words('english') + specific_stopwords + html_stopwords + common_words_to_ignore
    return stopwords


with open('./sklearn_kmeans_clustering.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    max_file_count = data['max_file_count']
    # model_file_name = data['model_file_name']
    cluster_count = data['cluster_count']
    random_state = data['random_state']

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
