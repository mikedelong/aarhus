import sys

import unittest
import aarhus
from aarhus.aarhus import custom_stopwords
import aarhus.aarhus.custom_stopwords
from nose.tools import *

def setup():
    print "SETUP!"

def teardown():
    print "TEAR DOWN!"

def test_basic():
    stopwords = custom_stopwords.stopwords()
    print (stopwords)

