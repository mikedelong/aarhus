import logging
import json
import elasticsearch
import numpy
from matplotlib import pyplot

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('elasticsearch_barchart_2.json') as data_file:
    data = json.load(data_file)
    logging.debug(data)
    elasticsearch_host = data['elasticsearch_host']
    elasticsearch_port = data['elasticsearch_port']
    elasticsearch_index_name = data['elasticsearch_index_name']
    elasticsearch_document_type = data['elasticsearch_document_type']
    search_term = data['search_term']

elasticsearch_server = elasticsearch.Elasticsearch([{'host': elasticsearch_host, 'port': elasticsearch_port}])

aggregation_name = 'kmeans_cluster'
t7 = {
    "size": 0,
    "query": {
        "bool": {
            "must": {
                "term": {
                    "body": search_term
                }
            }
        }
    },
    "aggregations": {
        aggregation_name: {
            "terms": {"field": "kmeans_cluster", "size": 60000}
        },
    }
}

t2 = elasticsearch_server.search(index=elasticsearch_index_name, doc_type=elasticsearch_document_type, body=t7)
logging.debug(t2)

total_hits = t2['hits']['total']
logging.debug('bucket count: %d, total hits: %d' % (len(t2['aggregations'][aggregation_name]['buckets']),
                                                    total_hits))

t4 = [float(bucket['doc_count']) / total_hits for bucket in t2['aggregations'][aggregation_name]['buckets']]
t5 = [int(bucket['key']) for bucket in t2['aggregations'][aggregation_name]['buckets']]

x_location = numpy.arange(len(t4))  # the x locations for the groups
bar_width = 0.5  # the width of the bars

figure, axis = pyplot.subplots()
rectangles = axis.bar(x_location, t4, bar_width, color='g')  # , yerr=menStd)

# add some text for labels, title and axes ticks
axis.set_ylabel('Hits (fraction)')
axis.set_xlabel('Cluster number')
axis.set_xticks(x_location + bar_width)
axis.set_xticklabels(tuple(t5))

pyplot.xticks(x_location, tuple(t5), rotation='vertical')

pyplot.show()
