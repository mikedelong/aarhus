import cPickle as pickle
import json
import logging
import os
import numpy
import sklearn.feature_extraction.text as text
from sklearn import decomposition

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('./matrix_factorization_input.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    input_folder = data['input_folder']
    pickle_file_name = data['pickle_file_name']
    max_file_count = data['max_file_count']
    topics_count = data['topics_count']
    top_words_count = data['top_words_count']

filenames = sorted([os.path.join(input_folder, file_name) for file_name in os.listdir(input_folder)])

# truncate
if max_file_count < len(filenames) and max_file_count != -1:
    filenames = filenames[:max_file_count]

# todo what is min_df
vectorizer = text.CountVectorizer(input='filename', stop_words='english', min_df=20, decode_error='ignore')
logging.debug('created vectorizer')
dtm = vectorizer.fit_transform(filenames).toarray()
logging.debug('created matrix')
vocabulary = numpy.array(vectorizer.get_feature_names())
logging.debug('matrix shape: %s, vocabulary size: %d', dtm.shape, len(vocabulary))
clf = decomposition.NMF(n_components=topics_count, random_state=0)
logging.debug('decomposition complete.')
doctopic = clf.fit_transform(dtm)
logging.debug('fit-transform complete.')
topic_words = []
for topic in clf.components_:
    word_idx = numpy.argsort(topic)[::-1][0:top_words_count]
    topic_words.append([vocabulary[word] for word in word_idx])
doctopic /= numpy.sum(doctopic, axis=1, keepdims=True)

names = []
for file_name in filenames:
    basename = os.path.basename(file_name)
    names.append(basename)

names = numpy.asarray(names)
doctopic_orig = doctopic.copy()
groups_count = len(set(names))
doctopic_grouped = numpy.zeros((groups_count, topics_count))

for i, name in enumerate(sorted(set(names))):
    doctopic_grouped[i, :] = numpy.mean(doctopic[names == name, :], axis=0)

doctopic = doctopic_grouped

t0 = sorted(set(names))

logging.info("Top NMF topics in...")

for i in range(len(doctopic)):
    top_topics = numpy.argsort(doctopic[i, :])[::-1][0:3]
    top_topics_str = ' '.join(str(t) for t in top_topics)
    # logging.info("{}: {}".format(names[i], top_topics_str))

for t in range(len(topic_words)):
    logging.info("Topic {}: {}".format(t, ' '.join(topic_words[t][:15])))

out_pickle = {
    'doctopic' : doctopic,
    'topic_words' : topic_words
}
pickle.dump(out_pickle, open( pickle_file_name, 'wb' ))
