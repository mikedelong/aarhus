import collections
import json
import logging
import pickle
import sys
import time

from matplotlib import pyplot as pyplot
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer('english')
stopwords = stopwords.words('english')


def get_character_sets(arg_message):
    charsets = set({})
    for character_set in arg_message.get_charsets():
        if character_set is not None:
            charsets.update([character_set])
    return charsets


def handle_error(arg_error_message, arg_email_message, arg_character_set):
    logging.warn('message: %s character_set: %s character sets found: %s subject: %s sender: %s' %
                 (arg_error_message, arg_character_set, get_character_sets(arg_email_message),
                  arg_email_message['subject'], arg_email_message['from']))


# https://stackoverflow.com/questions/7166922/extracting-the-body-of-an-email-from-mbox-file-decoding-it-to-plain-text-regard
def get_email_body(arg_message):
    body = None
    # Walk through the parts of the email to find the text body.
    if arg_message.is_multipart():
        for part in arg_message.walk():

            # If part is multipart, walk through the subparts.
            if part.is_multipart():

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True)

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif arg_message.get_content_type() == 'text/plain':
        body = arg_message.get_payload(decode=True)

        # No checking done to match the charset with the correct part.
    for charset in get_character_sets(arg_message):
        try:
            body = body.decode(charset)
        except UnicodeDecodeError:
            handle_error("UnicodeDecodeError: encountered.", arg_message, charset)
        except AttributeError:
            handle_error("AttributeError: encountered", arg_message, charset)
        except LookupError:
            handle_error("LookupError: encountered", arg_message, charset)
    return body


def run():
    start_time = time.time()
    logging.debug('started.')

    with open('roots-tfidf-settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
        limit = int(data['document_count_limit'])
        limit = sys.maxint if limit == -1 else limit
        input_pickle_file = data['input_pickle_file']
        kmeans_verbose = bool(data['k_means_verbose'])
        minibatch = bool(data['k_means_minibatch'])
        max_df = float(data['max_df'])
        min_df = float(data['min_df'])
        n_components = int(data['n_components'])
        n_features = int(data['n_features'])
        ngram_range_min = int(data['ngram_range_min'])
        ngram_range_max = int(data['ngram_range_max'])
        random_state = int(data['random_state'])
        terms_to_print = int(data['terms_to_print'])
        true_k = int(data['k_means_cluster_count'])
        use_idf = bool(data['tfidf_use_idf'])
        write_tfidf_vocabulary = data['write_tfidf_vocabulary']

    svd = TruncatedSVD(n_components, random_state=random_state)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    vectorizer_english = TfidfVectorizer(max_df=max_df, max_features=n_features, min_df=min_df,
                                         ngram_range=(ngram_range_min, ngram_range_max), stop_words='english',
                                         use_idf=use_idf)

    if minibatch:
        km = MiniBatchKMeans(batch_size=1000, init='k-means++', init_size=1000, n_clusters=true_k, n_init=1,
                             random_state=random_state, verbose=kmeans_verbose)
    else:
        km = KMeans(init='k-means++', max_iter=100, n_clusters=true_k, n_init=1, random_state=random_state,
                    verbose=kmeans_verbose)

    with open(input_pickle_file, 'rb') as input_fp:
        roots = pickle.load(input_fp)

    # http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
    logging.debug('After pickle load we have %d messages.' % len(roots))
    count = 0
    X = list()
    documents_processed = list()
    for key in roots.keys():
        value = roots[key]
        if count < limit:
            body = get_email_body(value)
            if body is not None:
                if False:
                    pass
                else:
                    try:
                        decoded = body.decode('utf-8', 'ignore')
                        X.append(decoded)
                        documents_processed.append(key)
                    except UnicodeDecodeError as unicodeDecodeError:
                        logging.warn(unicodeDecodeError)
                    pass
        count += 1

    logging.debug('After ignoring documents with unicode decode errors we have %d messages.' % len(X))
    original_size = (min(limit, len(roots)))
    actual_size = len(X)
    loss_percent = (100 * (original_size - actual_size) / original_size)
    logging.debug('We lost %d percent due to unicode errors: %d of %d' % (loss_percent, (original_size - actual_size),
                                                                          original_size))

    logging.debug('data extraction complete. Running TFIDF.')
    _ = vectorizer_english.fit_transform(X)
    logging.debug('The vocabulary contains %d words.' % len(vectorizer_english.vocabulary_.keys()))
    logging.debug('The model found %d stopwords.' % len(vectorizer_english.stop_words_))

    stopwords = vectorizer_english.stop_words_
    # todo move these to a data file
    stopwords.update(['http', 'https', 'com', 'org', 'mailto', 'www', 'unsubscription', 'edu', 'email_blast_key',
                      'googlegroups', 'jsp', 'rs6', 'constantcontact', 'gmail', 'unsubscribe', '3d', 'google',
                      'hangouts', 'link', 'safeunsubscribe', 'recipient', 'html', 'email', 'emails', 'iphone',
                      'subscribed', 'optout'])
    stopwords.update(['will', 'your', 'our', 'as', 'or', 'if', 'by', 'my', 'can', 'all', 'not', 'but', 'me',
                      'would', 'about', 'us', 'he', 'she', 'an', 'please', 'so', 'do', 'was', 'has', 'thanks',
                      'well', 'his', 've', 'what', 'who', 'just', 'know', 'call', 'sent', 'her', 'am', 'out',
                      'new', 'time', 'they', 'more', 'up', 'here', 'there', 'get', 'best', 'one', 're',
                      'their', 'now', 'let', 'any', 'the', 'need', 'work', 'good', 'hope', 'should', 'thank',
                      'how', 'have', 'been', 'no', 'could', 'also', 'make', 'its', 'some', 'may', 'think', 'when'])
    vectorizer_stopwords = TfidfVectorizer(max_df=max_df, max_features=n_features, min_df=min_df,
                                           ngram_range=(ngram_range_min, ngram_range_max),
                                           stop_words=stopwords, use_idf=use_idf)
    logging.debug('got the extended stopword list; rerunning TFIDF with the expanded list')
    X = vectorizer_stopwords.fit_transform(X)
    logging.debug('The vocabulary contains %d words.' % len(vectorizer_stopwords.vocabulary_.keys()))
    logging.debug('The model found %d stopwords.' % len(vectorizer_stopwords.stop_words_))

    logging.debug('TFIDF complete. About to start LSA.')

    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    logging.debug('with %d documents, %d components, and %d features we have %.2f explained variance.' %
                  (len(X), n_components, n_features, explained_variance))

    logging.debug("Clustering sparse data with %s" % km)
    km.fit(X)

    logging.debug("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    logging.debug("Top terms per cluster:")
    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]
    terms = vectorizer_stopwords.get_feature_names()
    for jndex in range(true_k):
        logging.debug('Cluster %d: %d : %s' % (
            jndex, km.counts_[jndex], [terms[index] for index in order_centroids[jndex, :terms_to_print]]))

    if write_tfidf_vocabulary:
        # todo move this to a setting
        tfidf_vocabulary_file = './roots_tfidf_vocabulary_out.csv'
        logging.debug('Writing tf-idf vocabulary to %s' % tfidf_vocabulary_file)
        with open(tfidf_vocabulary_file, 'wb') as output_fp:
            for key, value in vectorizer_stopwords.vocabulary_.iteritems():
                output_fp.write('%s,%d \n' % (key, value))

    logging.debug('lengths of labels: %d, documents processed: %d' % (len(km.labels_), len(documents_processed)))
    largest_cluster_number = collections.Counter(km.labels_).most_common(1)[0][0]
    largest_cluster = sorted([int(item[1]) for item in zip(km.labels_, documents_processed) if
                              item[0] == largest_cluster_number])
    logging.debug('largest cluster: %d (%d) : %s' % (largest_cluster_number, len(largest_cluster), largest_cluster))

    # use t-SNE to visualize
    model_tsne = TSNE(n_components=2, random_state=random_state)
    points_tsne = model_tsne.fit_transform(X)
    pyplot.figure()
    pyplot.scatter([each[0] for each in points_tsne], [each[1] for each in points_tsne])
    pyplot.show()
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
