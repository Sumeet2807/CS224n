#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CNN(nn.Module):
    
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self,c_embed_size,w_embed_size,kernel_size,pad):
    	super(CNN, self).__init__()
    	self.conv = torch.nn.Conv1d(c_embed_size,w_embed_size,kernel_size=kernel_size,padding=pad)
    	# self.max1d = torch.nn.MaxPool1d((word_size-4))

    def forward(self,x):

    	conv_out = torch.nn.functional.relu(self.conv(x))
    	embedding = torch.max(conv_out,dim=2).values
    	return(embedding) 


    ### END YOUR CODE

