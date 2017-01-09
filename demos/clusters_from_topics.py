# http://stats.stackexchange.com/questions/28904/how-to-cluster-lda-lsi-topics-generated-by-gensim
# coding:utf-8

import cPickle as pickle
import glob
import logging
import os
import scipy
import scipy.sparse
import string
import sys
import time
from collections import defaultdict

import gensim.matutils
import gensim.utils
import numpy
from gensim import corpora, models, similarities
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")

data_dir = os.path.join(os.getcwd(), 'data/')
output_dir = os.path.join(os.getcwd(), 'output/')
work_dir = os.path.join(os.getcwd(), 'model', os.path.basename(__file__).rstrip('.py'))
if not os.path.exists(work_dir):
    os.mkdir(work_dir)
os.chdir(work_dir)

logger = logging.getLogger('text_similar')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# convert to unicode
def to_unicode(arg_text):
    result = arg_text.lower()
    if not isinstance(result, unicode):
        result = result.decode('utf-8', 'ignore')
    result = ' '.join(
        ["".join([character for character in unicode(word) if character not in string.punctuation]) for word in
         result.split(' ') if not any([word.startswith('http:'), word.startswith('https:'),
                                       word.startswith('mailto:'), word.endswith('.com'),
                                       word.endswith('.org')])])
    return result


def to_unicode_unrolled(arg_text):
    t = arg_text.lower()
    result = []
    if not isinstance(t, unicode):
        t = t.decode('utf-8', 'ignore')
    for word in t.split(' '):
        b0 = word.startswith(u'http:')
        b6 = word.startswith(u'<http:')
        b1 = word.startswith(u'https:')
        b2 = word.startswith(u'mailto:')
        b3 = word.endswith(u'.com')
        b4 = word.endswith(u'.org')
        b5 = any([b0, b1, b2, b3, b4, b6])
        if not b5:
            word = ' '.join(
                ["".join([character for character in unicode(word) if character not in string.punctuation])])
        result.append(word)

    return " ".join(result)


def remove_stopwords_and_stem(arg_text):
    result = [stemmer.stem(item) for item in arg_text if item not in stopwords.words('english')]
    return result


class TextSimilar(gensim.utils.SaveLoad):
    def __init__(self):
        self.conf = {}
        self.dictionary = None
        self.docs = None
        self.fname = None
        self.lda = None
        self.lda_similarity_index = None
        self.lda_tfidf = None
        self.lda_tfidf_similarity_index = None
        self.logent = None
        self.logent_similarity_index = None
        self.lsi = None
        self.lsi_similarity_index = None
        self.method = None
        self.para = None
        self.similar_index = None
        self.tfidf = None

    def _preprocess(self):
        # todo write a more pythonic version of this function and use it
        docs = [to_unicode_unrolled(open(f, 'r').read().strip()).split() for f in glob.glob(self.fname)]
        logger.debug('ingested files into big array with length %d' % len(docs))
        docs = [remove_stopwords_and_stem(item) for item in docs]
        logger.debug('removed stopwords and stemmed')
        pickle.dump(docs, open(self.conf['fname_docs'], 'wb'))
        logger.debug('pickle dump to %s done' % self.conf['fname_docs'])

        dictionary = corpora.Dictionary(docs)
        dictionary.save(self.conf['fname_dict'])
        logger.debug('dictionary save to %s done' % self.conf['fname_dict'])

        corpus = [dictionary.doc2bow(doc) for doc in docs]
        corpora.MmCorpus.serialize(self.conf['fname_corpus'], corpus)
        logger.debug('corpus serialize to %s done' % self.conf['fname_corpus'])

        return docs, dictionary, corpus

    def _generate_conf(self):
        fname = self.fname[self.fname.rfind('/') + 1:]
        self.conf['fname_docs'] = '%s.docs' % fname
        self.conf['fname_dict'] = '%s.dict' % fname
        self.conf['fname_corpus'] = '%s.mm' % fname

    def train(self, arg_fname, is_pre=True, method='lsi', **params):
        self.fname = arg_fname
        self.method = method
        self._generate_conf()
        if is_pre:
            self.docs, self.dictionary, corpus = self._preprocess()
        else:
            self.docs = pickle.load(open(self.conf['fname_docs']))
            self.dictionary = corpora.Dictionary.load(self.conf['fname_dict'])
            corpus = corpora.MmCorpus(self.conf['fname_corpus'])

        if params is None:
            params = {}

        logger.info("training TF-IDF model")
        self.tfidf = models.TfidfModel(corpus, id2word=self.dictionary)
        corpus_tfidf = self.tfidf[corpus]

        if method == 'lsi':
            logger.info("training LSI model")
            self.lsi = models.LsiModel(corpus_tfidf, id2word=self.dictionary, **params)
            self.lsi.print_topics(-1)
            self.lsi_similarity_index = similarities.MatrixSimilarity(self.lsi[corpus_tfidf])
            self.para = self.lsi[corpus_tfidf]
        elif method == 'lda_tfidf':
            logger.info("training LDA model")
            # try 6 workers here instead of original 8
            self.lda_tfidf = models.LdaMulticore(corpus_tfidf, id2word=self.dictionary, workers=6, **params)
            self.lda_tfidf.print_topics(-1)
            self.lda_tfidf_similarity_index = similarities.MatrixSimilarity(self.lda[corpus_tfidf])
            self.para = self.lda[corpus_tfidf]
        elif method == 'lda':
            logger.info("training LDA model")
            # try 6 workers here instead of original 8
            self.lda = models.LdaMulticore(corpus, id2word=self.dictionary, workers=6, **params)
            self.lda.print_topics(-1)
            self.lda_similarity_index = similarities.MatrixSimilarity(self.lda[corpus])
            self.para = self.lda[corpus]
        elif method == 'logentropy':
            logger.info("training a log-entropy model")
            self.logent = models.LogEntropyModel(corpus, id2word=self.dictionary)
            self.logent_similarity_index = similarities.MatrixSimilarity(self.logent[corpus])
            self.para = self.logent[corpus]
        else:
            msg = "unknown semantic method %s" % method
            logger.error(msg)
            raise NotImplementedError(msg)

    def doc2vec(self, doc):
        bow = self.dictionary.doc2bow(to_unicode(doc).split())
        if self.method == 'lsi':
            return self.lsi[self.tfidf[bow]]
        elif self.method == 'lda':
            return self.lda[bow]
        elif self.method == 'lda_tfidf':
            return self.lda[self.tfidf[bow]]
        elif self.method == 'logentropy':
            return self.logent[bow]

    def find_similar(self, doc, n=10):
        vec = self.doc2vec(doc)
        sims = self.similar_index[vec]
        sims = sorted(enumerate(sims), key=lambda item: -item[1])
        for elem in sims[:n]:
            idx, value = elem
            print (' '.join(self.docs[idx]), value)

    def get_vectors(self):
        return self._get_vector(self.para)

    @staticmethod
    def _get_vector(corpus):

        def get_max_id():
            maxid = -1
            for document in corpus:
                maxid = max(maxid, max(
                    [-1] + [fieldid for fieldid, _ in document]))  # [-1] to avoid exceptions from max(empty)
            return maxid

        num_features = 1 + get_max_id()
        index = numpy.empty(shape=(len(corpus), num_features), dtype=numpy.float32)
        for docno, vector in enumerate(corpus):
            if docno % 1000 == 0:
                logger.info("PROGRESS: at document #%i/%i" % (docno, len(corpus)))

            if isinstance(vector, numpy.ndarray):
                pass
            elif scipy.sparse.issparse(vector):
                vector = vector.toarray().flatten()
            else:
                vector = gensim.matutils.unitvec(gensim.matutils.sparse2full(vector, num_features))
            index[docno] = vector

        return index


def cluster(vectors, ts, k=30, arg_method=None):
    from sklearn.cluster import k_means
    x = numpy.array(vectors)
    cluster_center, result, inertia = k_means(x.astype(numpy.float), n_clusters=k, init="k-means++")
    x__y_dic = defaultdict(set)
    for i, pred_y in enumerate(result):
        x__y_dic[pred_y].add(''.join(ts.docs[i]))

    logger.info ('len(x__y_dic): %d' % len(x__y_dic))
    output_file_name = arg_method + '-cluster.txt'
    with open(output_dir + output_file_name, 'w') as fo:
        for y in x__y_dic:
            fo.write(str() + '\n')
            fo.write('{word}\n'.format(word='\n'.join(list(x__y_dic[y])[:100])))


def main(arg_is_train=True):
    # todo make the data directory an input parameter
    # file_name = data_dir + '/files.tar'
    file_name = data_dir + '/*'

    # todo make this an input parameter
    topics_count = 100
    # todo make this an input parameter
    methods = ['lda', 'lda_tfidf', 'lsi'] # leaving out logentropy due to memory issues
    for method in methods:
        text_similar = TextSimilar()
        if arg_is_train:
            text_similar.train(file_name, method=method, num_topics=topics_count, is_pre=True, iterations=100)
            text_similar.save(method)
        else:
            text_similar = TextSimilar().load(method)

        index = text_similar.get_vectors()
        cluster(index, text_similar, k=topics_count, arg_method=method)


if __name__ == '__main__':
    is_train = True if len(sys.argv) > 1 else False
    start_time = time.time()
    main(is_train)
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
