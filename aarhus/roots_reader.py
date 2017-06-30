import json
import logging
import pickle
import sys
import time

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# http://mypy.pythonblogs.com/12_mypy/archive/1253_workaround_for_python_bug_ascii_codec_cant_encode_character_uxa0_in_position_111_ordinal_not_in_range128.html
reload(sys)
sys.setdefaultencoding("utf8")

logging.basicConfig(format='%(asctime)s : %(levelname)s :: %(message)s', level=logging.DEBUG)

stemmer = SnowballStemmer('english')
stopwords = stopwords.words('english')


def get_character_sets(arg_message):
    charsets = set({})
    for character_set in arg_message.get_charsets():
        if character_set is not None:
            charsets.update([character_set])
    return charsets


def handle_error(arg_error_message, arg_email_message, arg_character_set):
    logging.warn('message: %s character_set: %s character sets found: %s subject: %s sender: %s' %
                 (arg_error_message, arg_character_set, get_character_sets(arg_email_message),
                  arg_email_message['subject'], arg_email_message['from']))


# https://stackoverflow.com/questions/7166922/extracting-the-body-of-an-email-from-mbox-file-decoding-it-to-plain-text-regard
def get_email_body(arg_message):
    body = None
    # Walk through the parts of the email to find the text body.
    if arg_message.is_multipart():
        for part in arg_message.walk():

            # If part is multipart, walk through the subparts.
            if part.is_multipart():

                for subpart in part.walk():
                    if subpart.get_content_type() == 'text/plain':
                        # Get the subpart payload (i.e the message body)
                        body = subpart.get_payload(decode=True)

            # Part isn't multipart so get the email body
            elif part.get_content_type() == 'text/plain':
                body = part.get_payload(decode=True)

    # If this isn't a multi-part message then get the payload (i.e the message body)
    elif arg_message.get_content_type() == 'text/plain':
        body = arg_message.get_payload(decode=True)

        # No checking done to match the charset with the correct part.
    for charset in get_character_sets(arg_message):
        try:
            body = body.decode(charset)
        except UnicodeDecodeError:
            handle_error("UnicodeDecodeError: encountered.", arg_message, charset)
        except AttributeError:
            handle_error("AttributeError: encountered", arg_message, charset)
        except LookupError:
            handle_error("LookupError: encountered", arg_message, charset)
    return body


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
    limit = sys.maxint
    count = 0
    success = 0
    result = dict()
    for key in roots.keys():

        value = roots[key]
        if count < limit:
            body = get_email_body(value)
            if body is not None:
                success += 1
                try:
                    tokens = body.split()
                    tokens = [stemmer.stem(token).lower() for token in tokens]
                    tokens = [token for token in tokens if token not in stopwords]
                    logging.debug(len(tokens))
                    if len(tokens) == 0:
                        success -= 1
                    if len(tokens) >= 10:
                        result[key] = tokens
                except UnicodeDecodeError as unicodeDecodeError:
                    logging.warn(unicodeDecodeError)
                    success -= 1
                pass

        count += 1

    logging.debug('resulting tokens array has length %d' % len(result))
    # write out the tokens
    # todo move this to config
    output_pickle_file = './tokens.pickle'
    with open(output_pickle_file, 'wb') as output_fp:
        pickle.dump(result, output_fp)
    logging.debug('wrote %s' % output_pickle_file)
    logging.debug('%d %d %d' % (count, limit, success))
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logging.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))


if __name__ == '__main__':
    run()
