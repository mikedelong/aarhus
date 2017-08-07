import collections
import glob
import json
import logging
import pickle
import time
from os.path import basename

import nltk
import numpy
import pylab
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn import metrics
from sklearn.cluster import FeatureAgglomeration
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

start_time = time.time()
formatter = logging.Formatter('%(asctime)s : %(levelname)s :: %(message)s')

with open('pager-from-pickle-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
    input_folder = data['input_folder']
    # todo really should make sure this folder exists and create it if not
    output_folder = data['output_folder']

logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
# todo move log file name to setting
file_handler = logging.FileHandler(filename='./pager_from_pickle.log', delay=0, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
file_handler.setLevel(logging.DEBUG)
logger.debug('started.')

input_file = ''

problem_files = sorted(['digi330'])
some_files = sorted([
    '0084_LFeBk',
    '07-07-16-Voter-attitudes-release',
    '10.1.1.528.9390',
    '1603_DoubleDatabase_Codebook',
    '2016 Annual Report',
    '24649',
    '271339',
    '3_7_2017_tables',
    '58e7fef1935de',
    '958746-The-Pentagon-Papers-Volume-1-Gravel-Edition',
    '958764-The-Pentagon-Papers-Volume-2-Gravel-Edition',
    'a479311',
    'Benkler',
    'BUDGET-2017-BUD',
    'CPRT-115-HPRT-RU00-SAHR244-AMNT',
    'David_Foster_Wallace_-_Infinite_Jest',
    'donquixote',
    'GPO-WARRENCOMMISSIONREPORT',
    'Harvard-CAPS-Harris-Poll-April-Wave-Topline-Favorability-04.18.2017',
    'Moby-Dick',
    'Moby_Dick_NT',
    'Olson (1967) Logic of Collective Action (book)',
    'Progressive-Thinking',
    'Proust-1',
    'the-starfish-and-the-spider',
    'Ulysses'
])

planet_ebook_files = sorted(['1984',
                             'A-Christmas-Carol',
                             'A-Portrait-of-the-Artist-as-a-Young-Man',
                             'A-Tale-of-Two-Cities',
                             'Aesops-Fables',
                             'Agnes-Grey',
                             'Alices-Adventures-in-Wonderland',
                             'Andersens-Fairy-Tales',
                             'Anna-Karenina',
                             'Anne-of-Green-Gables',
                             'Around-the-World-in-80-Days',
                             'Beyond-Good-and-Evil',
                             'Bleak-House',
                             'Crime-and-Punishment',
                             'David-Copperfield',
                             'Down-and-Out-in-Paris-and-London',
                             'Dracula',
                             'Dubliners',
                             'Emma',
                             'Erewhon',
                             'For-the-Term-of-His-Natural-Life',
                             'Frankenstein',
                             'Great-Expectations',
                             'Grimms-Fairy-Tales',
                             'Gullivers-Travels',
                             'Heart-of-Darkness',
                             'Jane-Eyre',
                             'Kidnapped',
                             'Lady-Chatterlys-Lover',
                             'Les-Miserables',
                             'Little-Women',
                             'Madame-Bovary',
                             'Middlemarch',
                             'Moby-Dick',
                             'Northanger-Abbey',
                             'Nostromo-A-Tale-of-the-Seaboard',
                             'Notes-from-the-Underground',
                             'Of-Human-Bondage',
                             'Oliver-Twist',
                             'Paradise-Lost',
                             'Persuasion',
                             'Pollyanna',
                             'Pride-and-Prejudice',
                             'Robinson-Crusoe',
                             'Sense-and-Sensibility',
                             'Sons-and-Lovers',
                             'Swanns-Way',
                             'Tarzan-of-the-Apes',
                             'Tender-is-the-Night',
                             'Tess-of-the-dUrbervilles',
                             'The-Adventures-of-Huckleberry-Finn',
                             'The-Adventures-of-Tom-Sawyer',
                             'The-Brothers-Karamazov',
                             'The-Great-Gatsby',
                             'The-Hound-of-the-Baskervilles',
                             'The-Idiot',
                             'The-Illiad',
                             'The-Island-of-Doctor-Moreau',
                             'The-Jungle-Book',
                             'The-Last-of-the-Mohicans',
                             'The-Merry-Adventures-of-Robin-Hood',
                             'The-Metamorphosis',
                             'The-Odyssey',
                             'The-Picture-of-Dorian-Gray',
                             'The-Portrait-of-a-Lady',
                             'The-Prince',
                             'The-Scarlet-Pimpernel',
                             'The-Strange-Case-of-Dr-Jekyll',
                             'The-Thirty-Nine-Steps',
                             'The-Three-Musketeers',
                             'The-Time-Machine',
                             'The-Trial',
                             'Treasure-Island',
                             'Ulysses',
                             'Utopia',
                             'Vanity-Fair',
                             'War-and-Peace',
                             'Within-a-Budding-Grove',
                             'Women-In-Love',
                             'Wuthering-Heights'
                             ])

odyssey = 'The-Odyssey'
ulysses = 'Ulysses'
utopia = 'Utopia'

input_file_suffix = '.pickle'
output_file_suffix = '.png'

input_glob = input_folder + '*' + input_file_suffix
case_to_run = 5
if case_to_run < 0:
    pass
elif case_to_run == 0:
    files_to_process = [input_folder + each + input_file_suffix for each in some_files]
elif case_to_run == 1:
    files_to_process = glob.glob(input_glob)
elif case_to_run == 2:
    files_to_process = [input_folder + ulysses + input_file_suffix]
elif case_to_run == 3:
    files_to_process = [input_folder + utopia + input_file_suffix]
elif case_to_run == 4:
    files_to_process = [input_folder + each + input_file_suffix for each in planet_ebook_files]
elif case_to_run == 5:
    files_to_process = [input_folder + odyssey + input_file_suffix]
else:
    logger.warn('Case to run not defined; case to run is %d' % case_to_run)
    quit()

logger.debug('We have %d files to process.' % len(files_to_process))
for input_file_with_suffix in files_to_process:
    try:
        file_name_root = basename(input_file_with_suffix.replace(input_file_suffix, ''))
        input_file = input_folder + file_name_root + input_file_suffix
        logger.debug('input file is %s' % input_file)

        text = []
        with open(input_file, 'rb') as input_fp:
            input_object = pickle.load(input_fp)
            text = input_object['text']
            labels = input_object['page_numbers']
        logger.debug('our document has %d pages' % len(text))
        text = [each.replace('Free eBooks at Planet eBook.com', '') for each in text]
        lengths = [len(each) for each in text]
        logger.debug('page lengths:: min: %d max: %d mean: %d' % (min(lengths), max(lengths),
                                                                  sum(lengths) / len(lengths)))
        length_counts = collections.Counter(lengths)
        logger.debug('page length counts: %s' % sorted(length_counts.items()))

        use_idf = True
        n_features = 36500
        max_df = 0.95  # was 0.65
        min_df = 2  # was 3
        stop_words = 'english'
        # True False
        do_default = False
        do_porter_stemmed = False
        do_snowball_stemmed = False
        do_just_nouns = True
        if False:
            corpus = None
            text = None
        elif do_default:
            corpus = 'default'
        elif do_porter_stemmed:
            porter_stemmer = PorterStemmer()
            stemmed_text = []
            for each in text:
                tokens = each.split()
                line = ' '.join([porter_stemmer.stem(token.decode('utf-8')) for token in tokens])
                stemmed_text.append(line)
            text = stemmed_text
            corpus = 'porter'
        elif do_snowball_stemmed:
            snowball_stemmer = SnowballStemmer('english')
            stemmed_text = []
            for each in text:
                tokens = each.split()
                line = ' '.join([snowball_stemmer.stem(token.decode('utf-8')) for token in tokens])
                stemmed_text.append(line)
            text = stemmed_text
            corpus = 'snowball'
        elif do_just_nouns:
            noun_text = []
            for each in text:
                each = each.decode('utf-8')
                tokens = nltk.word_tokenize(each)
                tagged = nltk.pos_tag(tokens)
                line = ' '.join([item[0] for item in tagged if item[1][0] == 'N'])
                noun_text.append(line)
            text = noun_text
            corpus = 'nouns'

        tf_idf_vectorizer = TfidfVectorizer(max_df=max_df, max_features=n_features, min_df=min_df,
                                            stop_words=stop_words, use_idf=use_idf, ngram_range=(1, 3))
        tfidf = tf_idf_vectorizer.fit_transform(text)
        logger.debug('TF-IDF: n_samples: %d, n_features: %d, nnz: %d' % (tfidf.shape[0], tfidf.shape[1], tfidf.nnz))

        # todo make this a setting
        random_state = 2
        # todo make this a setting
        n_top_words = 10

        # todo make this a setting
        minibatch = False

        km = None
        for case in range(0, 3):
            if case == 0:
                do_lda = True
                do_lsa = False
                do_nmf = False
            elif case == 1:
                do_lda = False
                do_lsa = True
                do_nmf = False
            elif case == 2:
                do_lda = False
                do_lsa = False
                do_nmf = True

            topic_model_name = None

            if [do_lda, do_lsa, do_nmf].count(True) != 1:
                logger.warn('Can do exactly one of LDA, LSA, NMF.')
                quit()

            if False:
                pass
            elif do_lda:
                topic_model_name = 'LDA'
                tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=n_features, stop_words='english')
                tf = tf_vectorizer.fit_transform(text)
                # todo rationalize this
                true_k = tf.shape[0] * tf.shape[1] / tf.nnz
                logger.debug('using the TF/CountVectorizer data we expect %d clusters' % true_k)
                n_topics = true_k
                lda = LatentDirichletAllocation(learning_method='online', learning_offset=50., max_iter=5,
                                                n_topics=n_topics, random_state=random_state)
                lda.fit(tf)

                lda_results = lda.fit_transform(tf)
                tf_feature_names = tf_vectorizer.get_feature_names()
                for topic_idx, topic in enumerate(lda.components_):
                    logger.debug('Topic #%d:' % topic_idx)
                    logger.debug(
                        ' '.join(['[' + tf_feature_names[i] + ']' for i in topic.argsort()[:-n_top_words - 1:-1]]))
                # let's make a grid of topics and words
                if False:
                    values = lda.components_.copy()
                    # todo find a good threshold
                    # threshold = 0.75
                    # values[values < threshold] = 0
                    t4 = values.min()
                    t5 = values.max()
                    # todo find a way to filter out words with low importance
                    topic_data = [[tf_feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
                                  for topic_index, topic in enumerate(values)]
                    if False:
                        j3 = []
                        threshold = 0.0285
                        # todo choose the threshold so the topics meet some condition

                        for topic_index, topic in enumerate(values):
                            j0len = len(topic)
                            j0 = topic.argsort()[::-1][:j0len]
                            j6norm = numpy.linalg.norm(topic)
                            j6 = topic / j6norm
                            j6min = j6.min()
                            j6max = j6.max()
                            # j1 = [j6[i] for i in j0]
                            j2 = [tf_feature_names[i] for i in j0 if j6[i] > threshold]
                            logger.debug('(%d) : %s' % (len(j2), j2))
                            j3.append(j2)
                        pass
                    if False:
                        # this is the version that uses a percentile
                        # and it tends to produce consistent topic sample sizes
                        j3 = list()
                        for topic_index, topic in enumerate(values):
                            len_topic = len(topic)
                            j0 = numpy.percentile(topic, 99.7)
                            j1 = [i for i in range(0, len(topic)) if topic[i] > j0]
                            j2 = [tf_feature_names[i] for i in j1]
                            logger.debug('(%d) : %s' % (len(j2), j2))
                            j3.append(j2)
                    if False:
                        # this is the version that uses threshold-filtered data to choose words
                        # for values above a certain threshold
                        # but it tends to produce wildly varing topic sizes
                        j3 = []
                        threshold = 0.7
                        for topic_index, topic in enumerate(values):
                            j0 = topic.argsort()  # [:-n_top_words - 1:-1]
                            j1 = [topic[i] for i in j0]
                            j2 = [tf_feature_names[i] for i in j0 if values[topic_index, i] > threshold]
                            j3.append(j2)

                    # topic_data = [[topic_data[t][i] if values[t, i] > 0] for t, topic in enumerate(topic_data) for i, item in enumerate(topic_data[t]) ]

                    all_words = [each for every in topic_data for each in every]
                    # remove duplicates
                    t2 = []
                    [t2.append(word) for word in all_words if not t2.count(word)]
                    all_words = t2
                    ys = range(0, len(topic_data))
                    xs = range(0, len(all_words))
                    if True:
                        topic_mesh = numpy.zeros((len(topic_data), len(all_words)), dtype=numpy.int)
                        for i0, t0 in enumerate(topic_data):
                            for each in topic_data[i0]:
                                i1 = all_words.index(each)
                                j1 = tf_feature_names.index(each)
                                value = values[i0, j1]

                                topic_mesh[i0, i1] = value
                    else:
                        shape = lda.components_.shape
                        topic_mesh = numpy.zeros(shape, dtype=numpy.int)
                        for j0 in range(0, lda.components_.shape[0]):
                            for j1 in range(0, lda.components_.shape[1]):
                                value = lda.components_[j0, j1]
                                topic_mesh[j0, j1] = value if value > 0.001 else 0

                    pylab.yticks(ys, [str(item) for item in ys], fontsize=6)
                    pylab.xticks(range(0, len(xs)), all_words, fontsize=6, rotation='vertical')
                    pylab.imshow(topic_mesh, cmap='binary')
                    out_file = output_folder + file_name_root + '-' + topic_model_name + '-' + 'topics' + \
                               output_file_suffix
                    logger.debug('writing topic picture to output file %s' % out_file)
                    pylab.savefig(out_file)
                    pylab.clf()

                true_k = tf.shape[0] * tf.shape[1] / tf.nnz
                logger.debug('we are looking for %d clusters' % true_k)
                verbose = False
                if minibatch:
                    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000,
                                         verbose=verbose, random_state=random_state, max_iter=1000,
                                         reassignment_ratio=0.001,
                                         max_no_improvement=100)
                else:
                    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=verbose,
                                random_state=random_state)
                logger.debug("Clustering sparse data with %s" % km)

                km.fit(lda_results)
                ticks = sorted(set(km.labels_))
                logger.debug('here are our ticks: %s' % ticks)
                # todo add metrics to report
                homogeneity = metrics.homogeneity_score(labels, km.labels_)
                silhouette_score = metrics.silhouette_score(lda_results, km.labels_, sample_size=1000,
                                                            random_state=random_state)
                logger.debug("Silhouette Coefficient: %0.3f" % silhouette_score)
                logger.debug("Homogeneity: %0.3f" % homogeneity)
                text_to_display = 'homogeneity: %.2f\nsilhouette: %.2f' % (homogeneity, silhouette_score)
            elif do_lsa:
                topic_model_name = 'LSA'
                n_components = 300
                n_components = min(100, tfidf.shape[1] - 1)

                svd = TruncatedSVD(n_components, random_state=random_state)
                normalizer = Normalizer(copy=False)
                lsa = make_pipeline(svd, normalizer)
                lsa_results = lsa.fit_transform(tfidf)
                explained_variance = svd.explained_variance_ratio_.sum()
                logger.debug('Explained variance of the SVD step: %d' % int(explained_variance * 100))

                true_k = tfidf.shape[0] * tfidf.shape[1] / tfidf.nnz
                logger.debug('we are looking for %d clusters' % true_k)
                verbose = False
                if minibatch:
                    km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1, init_size=1000, batch_size=1000,
                                         verbose=verbose, random_state=random_state, max_iter=1000,
                                         reassignment_ratio=0.001,
                                         max_no_improvement=100)
                else:
                    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, verbose=verbose,
                                random_state=random_state)
                logger.debug("Clustering sparse data with %s" % km)

                km.fit(lsa_results)
                ticks = sorted(set(km.labels_))
                logger.debug('here are our ticks: %s' % ticks)
                homogeneity = metrics.homogeneity_score(labels, km.labels_)
                logger.debug("Homogeneity: %0.3f" % homogeneity)
                completeness = metrics.completeness_score(labels, km.labels_)
                logger.debug("Completeness: %0.3f" % completeness)
                v_measure = metrics.v_measure_score(labels, km.labels_)
                logger.debug("V-measure: %0.3f" % v_measure)
                adjusted_rand_score = metrics.adjusted_rand_score(labels, km.labels_)
                logger.debug("Adjusted Rand-Index: %.3f" % adjusted_rand_score)
                silhouette_score = metrics.silhouette_score(lsa_results, km.labels_, sample_size=1000,
                                                            random_state=random_state)
                logger.debug("Silhouette Coefficient: %0.3f" % silhouette_score)

                logger.debug("Top terms per cluster:")
                original_space_centroids = svd.inverse_transform(km.cluster_centers_)
                order_centroids = original_space_centroids.argsort()[:, ::-1]

                terms = tf_idf_vectorizer.get_feature_names()
                for i in range(true_k):
                    logger.debug("Cluster %d:" % i)
                    # cluster_max = min(n_top_words, len(terms[i]))

                    logger.debug([ind for ind in order_centroids[i, :n_top_words]])  # was n_top_words
                    logger.debug(
                        ['[' + terms[ind] + ']' for ind in order_centroids[i, :n_top_words]])  # was n_top_words
                text_to_display = 'explained variance: %.2f\nhomogeneity: %.2f\ncompleteness: %.2f\nv-measure: %.2f\n' \
                                  'adjusted rand: %.2f\nsilhouette: %.2f' % \
                                  (explained_variance, homogeneity, completeness, v_measure, adjusted_rand_score,
                                   silhouette_score)
            elif do_nmf:
                n_topics = true_k
                topic_model_name = 'NMF'
                nmf = NMF(n_components=n_topics, alpha=.1, l1_ratio=.5, random_state=random_state)
                nmf_results = nmf.fit_transform(tfidf)
                tfidf_feature_names = tf_idf_vectorizer.get_feature_names()
                for topic_idx, topic in enumerate(nmf.components_):
                    logger.debug('Topic #%d:' % topic_idx)
                    logger.debug(
                        ' '.join(['[' + tfidf_feature_names[i] + ']' for i in topic.argsort()[:-n_top_words - 1:-1]]))
                km.fit(nmf_results)
                ticks = sorted(set(km.labels_))
                logger.debug('here are our ticks: %s' % ticks)
                homogeneity = metrics.homogeneity_score(labels, km.labels_)
                silhouette_score = metrics.silhouette_score(lsa_results, km.labels_, sample_size=1000,
                                                            random_state=random_state)
                logger.debug("Silhouette Coefficient: %0.3f" % silhouette_score)
                logger.debug("Homogeneity: %0.3f" % homogeneity)
                text_to_display = 'homogeneity: %.2f\nsilhouette: %.2f' % (homogeneity, silhouette_score)

            else:
                logger.warn('no topic model chosen; quitting.')
                quit()

            # True False
            do_agglomeration = False
            do_pca = False
            do_tsne = True
            if [do_agglomeration, do_pca, do_tsne].count(True) != 1:
                logger.warn('Can do exactly one of FeatureAgglomeration, PCA, t-SNE.')
                quit()
 
            # todo add a test to make sure we do one of these
            if False:
                pass
            elif do_agglomeration:
                model = FeatureAgglomeration()
                if False:
                    pass
                elif do_lda:
                    try:
                        scatter_points = model.fit_transform(lda_results)
                    except AssertionError as assertionError:
                        logger.warn('%s : %s', (input_file, assertionError))
                elif do_lsa:
                    scatter_points = model.fit_transform(lsa_results)
                elif do_nmf:
                    scatter_points = model.fit_transform(nmf_results)
            elif do_pca:
                pass
                model = PCA(n_components=2, random_state=random_state)
                if False:
                    pass
                elif do_lda:
                    try:
                        scatter_points = model.fit_transform(lda_results)
                    except AssertionError as assertionError:
                        logger.warn('%s : %s', (input_file, assertionError))
                elif do_lsa:
                    scatter_points = model.fit_transform(lsa_results)
                elif do_nmf:
                    scatter_points = model.fit_transform(nmf_results)

                pca_explained_variance = model.explained_variance_
                logger.debug('PCA explained variance: %s' % pca_explained_variance)
                pca_text = '\nPCA explained variance: %s' % pca_explained_variance
                text_to_display += pca_text

            elif do_tsne:
                tsne_init = 'random'  # could also be 'pca'
                tsne_init = 'pca'
                # we would really like this to be related to the centroids from the k-means
                # todo initialize with the k-means centroids

                tsne_perplexity = 20.0
                tsne_early_exaggeration = 4.0
                tsne_learning_rate = 300  # was1000.0
                model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=tsne_perplexity,
                             early_exaggeration=tsne_early_exaggeration, learning_rate=tsne_learning_rate)
                if False:
                    pass
                elif do_lda:
                    try:
                        scatter_points = model.fit_transform(lda_results)
                    except AssertionError as assertionError:
                        logger.warn('%s : %s', (input_file, assertionError))
                elif do_lsa:
                    scatter_points = model.fit_transform(lsa_results)
                elif do_nmf:
                    scatter_points = model.fit_transform(nmf_results)

                kl_divergence = model.kl_divergence_
                logger.debug('KL divergence: %.2f' % kl_divergence)
                kl_divergence_text = '\nKL divergence: %.2f' % kl_divergence
                text_to_display += kl_divergence_text

                logger.debug('finished TSNE')

            colormap = 'plasma'  # was 'gnuplot'
            figsize = (16, 9)
            pylab.figure(figsize=figsize)
            # todo remove these two lines
            xs = numpy.array([each[0] for each in scatter_points])
            ys = numpy.array([each[1] for each in scatter_points])

            clusters = numpy.array(km.labels_)
            if False:
                marker_choices = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x',
                                  'D', 'd']
                # markers = itertools.cycle(marker_choices)
                x1 = len(marker_choices)
                different_labels = set(km.labels_)
                marker_count = min(10, len(marker_choices))
                x0 = len(different_labels) / marker_count
                colors = pylab.cm.plasma(numpy.linspace(0, 1, x0))
                # todo use a numpy array with a filter
                for index, marker in enumerate(marker_choices[:marker_count]):
                    pylab.scatter(xs[clusters == index], ys[clusters == index], marker=marker, c=colors)

            if False:
                pylab.scatter(xs, ys, marker='x', c=km.labels_, cmap=colormap)
            else:
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
                marker_choices = ['.', 'o', 'v', 's', 'x', 'D', '^']
                kj = len(colors)
                different_labels = sorted(set(km.labels_))
                kl = len(different_labels) / kj + 1
                for kk in different_labels:
                    kk0 = kk / kl
                    if kk0 > len(marker_choices):
                        pass
                    marker = marker_choices[kk / kl]
                    color = colors[kk % kj]
                    pylab.scatter(xs[clusters == kk], ys[clusters == kk], marker=marker, color=color)

            index = 1
            if False:
                for x, y in zip(xs, ys):
                    # for the moment let's not add the page numbers
                    pylab.text(x, y, str(index), color='k', fontsize=6)
                    index += 1
            # pylab.margins(0.1)
            title = ' '.join([basename(file_name_root), 'pages: ', str(len(text)), 'clusters:', str(true_k)])
            pylab.title(title)
            # todo make the location of this text box sensible
            pylab.text(1.5 * min(xs), 0, text_to_display, fontsize=12)
            ticks = sorted(set(km.labels_))
            if False:
                color_bar = pylab.colorbar(ticks=ticks)
                color_bar.ax.tick_params(labelsize=7)
                # todo add cluster sizes to the ticks
                color_bar.set_ticklabels(ticks)
            out_file = output_folder + file_name_root + '-' + corpus + '-' + topic_model_name + output_file_suffix
            logger.debug('writing figure to output file %s' % out_file)
            pylab.savefig(out_file)
    except ValueError as valueError:
        logger.warn('%s : %s' % (input_file, valueError))

logger.debug('done')
elapsed_time = time.time() - start_time
logger.debug('elapsed time %d seconds', elapsed_time)
