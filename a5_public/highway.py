#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class Highway(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f

    def __init__(self,word_dim):
    	super(Highway, self).__init__()

    	self.proj = torch.nn.Linear(word_dim,word_dim)
    	self.gate = torch.nn.Linear(word_dim,word_dim)
    	self.dropout = torch.nn.Dropout()

    def forward(self,x):

    	proj = torch.nn.functional.relu(self.proj(x))
    	gate = torch.sigmoid(self.gate(x))
    	return(self.dropout((proj * gate) + ((1-gate) * x))) 


    ### END YOUR CODE

