
import logging


class Importer(object):

    logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.INFO)

def run():
    instance = Importer()
    pass

if __name__ == '__main__':
    run()

