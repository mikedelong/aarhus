
import logging
import pickle
import pylab

def do(index):
    figsize = (16, 9)
    pylab.figure(figsize=figsize)
    real_index = 2 * index
    xs = [each[0] for each in items[real_index]['points']]
    ys = [each[1] for each in items[real_index]['points']]
    clusters = [each for each in items[real_index]['clusters']]
    colormap = 'plasma'
    pylab.scatter(xs, ys, marker='x', c=clusters, cmap=colormap)
    pylab.savefig('./' + str(real_index) + '.png')
    pylab.clf()


formatter = logging.Formatter('%(asctime)s : %(levelname)s :: %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
# todo move log file name to setting
log_file_name = './tsne_pickle_reader.log'
# file_handler = logging.FileHandler(filename=log_file_name, delay=0, mode='w')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)
# file_handler.setLevel(logging.DEBUG)
logger.debug('started.')

input_file = './tsne_results.pickle'
logger.debug('reading data from %s' % input_file)
with open(input_file, 'rb') as input_fp:
    items = pickle.load(input_fp)
    logger.debug(len(items))
    logger.debug(items.keys())

for index in sorted(items.keys()):
    logger.debug('%d %d' % (index, len(items[index])))
    current = items[index]
    points = current['points']
    cluster = current['clusters']
    do(index/2)



