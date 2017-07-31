import glob
import json
import logging
import pickle
import time
from cStringIO import StringIO
from os.path import basename
from os.path import isfile

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pyPdf import PdfFileReader


# note that this version returns a list of strings, one string per page
def convert_pdf_to_text(arg_path, arg_codec='utf-8', arg_verbose=False):
    resource_manager = PDFResourceManager()
    result_string = StringIO()
    laparams = LAParams()
    device = TextConverter(resource_manager, result_string, codec=arg_codec, laparams=laparams)
    file_pointer = file(arg_path, 'rb')
    interpreter = PDFPageInterpreter(resource_manager, device)
    password = ""
    max_pages = 0
    caching = True
    unused = set()
    page_numbers = list()
    page_count = 0
    page_markers = [0]
    value = None
    for page in PDFPage.get_pages(file_pointer, unused, maxpages=max_pages, password=password, caching=caching,
                                  check_extractable=True):
        process_page_t0 = time.time()
        interpreter.process_page(page)
        process_page_t1 = time.time()
        page_numbers.append(device.cur_item.pageid)
        page_count += 1
        if arg_verbose:
            logging.debug('processed page %d in %.2f seconds' % (page_count, process_page_t1 - process_page_t0))
        value = result_string.getvalue()
        page_markers.append(len(value))
    file_pointer.close()
    device.close()
    result_string.close()
    text_result = [value[page_markers[page_index]:page_markers[page_index + 1]] for page_index in
                   range(0, len(page_markers) - 1)]
    result = {'page_numbers': page_numbers, 'text': text_result}
    return result


start_time = time.time()
logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

with open('pdf-extract-settings.json') as data_file:
    data = json.load(data_file)
    logging.debug(json.dumps(data, sort_keys=True, indent=4, separators=(',', ': ')))
    destination_folder = data['destination_folder']
    documents_root_folder = data['documents_root_folder']
    downloads_root_folder = data['downloads_root_folder']

documents_pdf_glob = glob.glob(documents_root_folder + '*.pdf')
downloads_pdf_glob = glob.glob(downloads_root_folder + '*.pdf')
files_to_process = list()
files_to_process.extend(downloads_pdf_glob)
# files_to_process.extend(documents_pdf_glob)
count = 0
threshold = 75
for file_name in files_to_process:
    # check to see if the destination file exists
    # todo check file creation datetime if the file exists
    output_short_name = basename(file_name)[0:-4]
    output_file = destination_folder + output_short_name + '.pickle'
    if isfile(output_file):
        logging.warn('Output file %s exists; skipping.' % output_file)
    else:
        with open(file_name, 'rb') as input_fp:
            try:
                pdf = PdfFileReader(input_fp)
                number_of_pages = pdf.getNumPages()
                if number_of_pages > threshold:
                    count += 1
                    logging.debug('%d %d : %s' % (count, number_of_pages, file_name))
                    text = convert_pdf_to_text(arg_path=file_name, arg_verbose=True)
                    with open(output_file, 'wb') as pickle_fp:
                        pickle.dump(text, pickle_fp)
            except AssertionError as assertionError:
                logging.warn(assertionError)
            except Exception as exception:
                logging.warn(exception)

elapsed_time = time.time() - start_time
logging.debug('elapsed time %d seconds', elapsed_time)
