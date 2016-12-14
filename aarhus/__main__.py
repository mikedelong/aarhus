
import json
import logging


class Importer(object):

    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

    def __init__(self, arg_input_folder):
        logging.debug(arg_input_folder)


def run():
    with open('settings.json') as data_file:
        data = json.load(data_file)
        logging.debug(data)
        input_folder = data['input_folder']

    instance = Importer(input_folder)
    pass

if __name__ == '__main__':
    run()

