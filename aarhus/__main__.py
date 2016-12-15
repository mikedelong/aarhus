import json
import logging
import os
import time


class Importer(object):
    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

    def __init__(self):
        pass

    def process_folder(self, arg_folder):
        document_count = 0
        for root, subdirectories, files in os.walk(arg_folder):
            for current in files:
                current_full_file_name = os.path.join(root, current)
                logging.debug("%d %s", document_count, current_full_file_name)


def run():
    start_time = time.time()

    with open('settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        input_folder = data['input_folder']

    instance = Importer()
    instance.process_folder(input_folder)

    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info(
        "Elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
