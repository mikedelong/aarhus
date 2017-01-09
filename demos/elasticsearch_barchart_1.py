import json
import logging

import elasticsearch
import numpy
from matplotlib import pyplot

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('elasticsearch_barchart_1.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    elasticsearch_host = data['elasticsearch_host']
    elasticsearch_port = data['elasticsearch_port']
    elasticsearch_index_name = data['elasticsearch_index_name']
    elasticsearch_document_type = data['elasticsearch_document_type']

elasticsearch_server = elasticsearch.Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])

aggregation_name = "clusters"
t0 = {
    "aggregations": {
        aggregation_name: {
            "histogram": {
                "field": "kmeans_cluster",
                "interval": 1,
                "order": {"_count": "desc"}
            }
        }
    }
}

t2 = elasticsearch_server.search(index=elasticsearch_index_name, doc_type=elasticsearch_document_type, body=t0)
logging.debug(t2)

t4 = [bucket['doc_count'] for bucket in t2['aggregations'][aggregation_name]['buckets']]
t5 = [int(bucket['key']) for bucket in t2['aggregations'][aggregation_name]['buckets']]

x_location = numpy.arange(len(t4))  # the x locations for the groups
bar_width = 0.35  # the width of the bars

figure, axis = pyplot.subplots()
rectangles = axis.bar(x_location, t4, bar_width, color='g')  # , yerr=menStd)

# add some text for labels, title and axes ticks
axis.set_ylabel('Cluster size')
axis.set_xlabel('Cluster number')
axis.set_xticks(x_location + bar_width)
axis.set_xticklabels(tuple(t5))

pyplot.xticks(x_location, tuple(t5), rotation='vertical')

pyplot.show()
