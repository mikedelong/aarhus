import unittest

from aarhus.aarhus import custom_stopwords


class TestStopwords(unittest.TestCase):
    def test_basic(self):
        stopwords = custom_stopwords.get_stopwords()
        print (stopwords)


if __name__ == '__main__':
    unittest.main()
