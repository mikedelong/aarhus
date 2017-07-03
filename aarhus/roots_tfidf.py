import json
import logging
import pickle
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
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
        logging.debug(data)
        input_pickle_file = data['input_pickle_file']
        output_pickle_file = data['output_pickle_file']
        min_df = float(data['min_df'])
        max_df = float(data['max_df'])

    n_components = 1200
    svd = TruncatedSVD(n_components)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    n_features = 200000
    use_idf = True
    vectorizer = TfidfVectorizer(max_df=max_df, max_features=n_features, min_df=min_df, stop_words='english',
                                 use_idf=use_idf)

    with open(input_pickle_file, 'rb') as input_fp:
        roots = pickle.load(input_fp)

    # http://scikit-learn.org/stable/auto_examples/text/document_clustering.html
    logging.debug('we have %d messages.' % len(roots))
    limit = sys.maxint
    limit = 5000
    count = 0
    success = 0
    result = dict()
    X = list()
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
                    except UnicodeDecodeError as unicodeDecodeError:
                        logging.warn(unicodeDecodeError)
                    pass

        count += 1

    logging.debug('data extraction complete. About to start TFIDF.')
    X = vectorizer.fit_transform(X)
    logging.debug('TFIDF complete. About to start LSA.')

    X = lsa.fit_transform(X)
    explained_variance = svd.explained_variance_ratio_.sum()
    logging.debug('with %d documents, %d components, and %d features we have %.2f explained variance.' %
                  (len(X), n_components, n_features, explained_variance))

    minibatch = True
    true_k = 20
    verbose = True
    if minibatch:
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=verbose)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,verbose=verbose)

    logging.debug("Clustering sparse data with %s" % km)
    km.fit(X)
    # todo find a way to associate message names with clusters
    # logging.debug("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # logging.debug("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # logging.debug("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # logging.debug("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    logging.debug("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, km.labels_, sample_size=1000))
    logging.debug("Top terms per cluster:")

    original_space_centroids = svd.inverse_transform(km.cluster_centers_)
    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        logging.debug('Cluster %d: %s' % (i, [terms[index] for index in order_centroids[i, :10]]))
    if False:
        # write out the tokens
        logging.debug('resulting tokens array has length %d' % len(result))
        with open(output_pickle_file, 'wb') as output_fp:
            pickle.dump(result, output_fp)
        logging.debug('wrote %s' % output_pickle_file)
        logging.debug('%d %d %d' % (count, limit, success))

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
