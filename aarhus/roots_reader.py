import json
import logging
import pickle
import time

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)


def run():
    start_time = time.time()
    logging.debug('started.')

    with open('roots-settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        pickle_file = data['pickle_file']

    with open(pickle_file, 'rb') as input_fp:
        roots = pickle.load(input_fp)

    logging.debug('we have %d messages.' % len(roots))

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
