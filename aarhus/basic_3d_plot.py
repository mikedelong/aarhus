# https://stackoverflow.com/questions/36438857/how-visualize-output-cluster-with-each-cluster-unique-colors
import numpy
from matplotlib import style
from matplotlib import pyplot

style.use('ggplot')
from sklearn.cluster import KMeans
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
# import pylab

import logging

formatter = logging.Formatter('%(asctime)s : %(levelname)s :: %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)

numpy.random.seed(1)

X = numpy.array([[1, 2, 5],
                 [5, 8, 2],
                 [1.5, 1.8, 6],
                 [8, 8, 9],
                 [1, 0.6, 10],
                 [2.5, 3.8, 6],
                 [2.5, 5.8, 9],
                 [5, 8, 3],
                 [4, 0.6, 7],
                 [2.5, 1.8, 4.6],
                 [6.5, 1.8, 12],
                 [7, 8, 9],
                 [2, 0.6, 7],
                 [5.5, 1.8, 4],
                 [4.8, 6.9, 6],
                 [4.9, 9.8, 2],
                 [9, 11, 12]])

cluster_num = 3
kmeans = KMeans(n_clusters=cluster_num)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
logger.debug('centroids: %s' % centroids)
logger.debug('labels: %s' % labels)
color = ['r', 'g', 'b']
c = Counter(labels)
fig = pyplot.figure()
ax = fig.gca(projection='3d')
for i in range(len(X)):
    logger.debug('coordinate: %s label: %s' % (X[i], labels[i]))
    logger.debug('i : %d' % i)
    logger.debug('color[labels[i]] : %s ' % color[labels[i]])
    ax.scatter(X[i][0], X[i][1], X[i][2], c=color[labels[i]])

for cluster_number in range(cluster_num):
    logger.debug('Cluster {} contains {} samples'.format(cluster_number, c[cluster_number]))

ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='x', s=150, linewidths=5, zorder=100, c=color)
pyplot.show()
