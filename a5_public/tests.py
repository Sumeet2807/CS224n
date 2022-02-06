import json
import math
import pickle
import sys
import time

import numpy as np

from docopt import docopt
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import pad_sents_char, batch_iter, read_corpus
from vocab import Vocab, VocabEntry

from char_decoder import CharDecoder
from nmt_model import NMT


import torch
import torch.nn as nn
import torch.nn.utils

import cnn


def test_conv():

	

	word_size = 15
	c_embed_size = 8
	w_embed_size = 32
	sentence_size = 20

	x = torch.ones([sentence_size,c_embed_size,word_size])
	model = cnn.CNN(word_size,c_embed_size,w_embed_size)
	out = model(x)
	print('expected_shape' + '---' + str(sentence_size) + ',' + str(w_embed_size))
	print('*******')
	print(out.shape)



print('running test')
test_conv()

# print('hello !')