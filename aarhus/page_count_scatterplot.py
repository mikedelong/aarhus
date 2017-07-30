import glob
import logging
import math
import pickle
import random
import time
import json
import matplotlib.pyplot as pyplot

start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)
logging.debug('started.')

with open('page-count-scatterplot-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
    input_folder = data['input_folder']
    input_file_list = data['input_file_list']
    output_folder = data['output_folder']
    random_seed = int(data['random_seed'])

random.seed(random_seed)
file_suffix = '.pickle'
glob_pattern = input_folder + '*' + file_suffix

files_to_process = list()
# True False
do_all_files = False
do_planet_ebook_files = True

planet_ebook_files = None
with open(input_file_list, 'rb') as input_fp:
    planet_ebook_files = input_fp.readlines()
planet_ebook_files = [each.strip() for each in planet_ebook_files]
logging.debug('we have %d input files to process' % len(planet_ebook_files))

if False:
    pass
elif do_all_files:
    files_to_process = glob.glob(glob_pattern)
elif do_planet_ebook_files:
    files_to_process = [input_folder + each + file_suffix for each in planet_ebook_files]
    logging.debug('files: %s' % files_to_process)

names = list()
counts = list()
log_counts = list()
for input_file, short_name in zip(files_to_process, planet_ebook_files):
    logging.debug(input_file)
    with open(input_file, 'rb') as input_fp:
        item = pickle.load(input_fp)
        counts.append(len(item))
        log_counts.append(math.log(len(item)))
        names.append(short_name.replace('-', ' ') + ' (' + str(len(item)) + ')')

# sort the data by the page counts
t0 = sorted(zip(counts, names))
counts = zip(*t0)[0]
names = zip(*t0)[1]

pyplot.figure(figsize=(16, 9))
indexes = range(0, len(counts))
pyplot.scatter(indexes, indexes, marker='x')

for index, name in enumerate(names):
    pyplot.annotate(name, (index, index), fontsize=8)

pyplot.tight_layout()
output_file_suffix = '.png'
out_file = output_folder + 'planet_ebook_file_size_plot' + output_file_suffix
logging.debug('writing figure to output file %s' % out_file)
pyplot.savefig(out_file)

logging.debug('done')
elapsed_time = time.time() - start_time
logging.debug('elapsed time %d seconds', elapsed_time)
pyplot.show()
