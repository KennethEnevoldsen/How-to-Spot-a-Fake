#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:34:39 2017

@author: kennethenevoldsen

Text Mining Module from tmgu2017
"""
    #Import modules
import io, glob, re
import pandas as pd

#########################################

    #Defining functions
def readtxt(filepath):
    """readtxt() extract text from file in filepath"""
    with io.open(filepath, 'r', encoding = 'utf-8') as f:
        content = f.read()
    return content

def read_dir_text(dir_path):
    """read all files in dir_path and puts them into list"""
    filepaths = glob.glob(dir_path + '*.txt')
    print filepaths #this is actually unecessary
    result_list = []
    for filepath in filepaths:
        text = readtxt(filepath)
        result_list.append(text)
    return result_list

def tokenize(text, lentoken = 0):
    """tokenize lowercase and tokenize the text"""
    tokenizer = re.compile(r'[^a-zA-Z]+')
    tokens = [token.lower() for token in tokenizer.split(text) if len(token) > lentoken]
    return tokens

def tf(tokenlist): #creates a dictionary containing all the words and their frequency
    tf = dict([(token, tokenlist.count(token)) for token in set(tokenlist)])
    return tf

def slice_tokens(tokens, n = 100, cut_off = True):
    """slice tokenized in slices of n tokens
    The cut_off means that it cut off the last piece which most likely will
    not have the same length as the rest"""
    slices = []    #result list of slices
    for i in range(0,len(tokens), n):  #slice tokens in accordance with n
        slices.append(tokens[i:(i+n)])
    if cut_off:
        del slices[-1] #delete the last slice
    return slices

    #Defining LabMT sentiment analyzer

def labMT_sent(text, rel = False, lentoken = 0):
    """labMT_sent uses LabMT's dictionary for sentiment analysis and returns a
    sentiment score when input a text.
    rel stands for relative (if the sentiment is relative to the frequency or not)
    lentoken is the len of tokens which should be removed (by deffault is removed 0 length tokens)
    """

    #clean text
    text = re.sub(r'[^a-zA-Z]', ' ', text) #replace everything that is not alfapethical with a space
    text = re.sub(r' +', ' ', text) #replace one or more whitespaces with a whitespace
    text = text.rstrip() #remove newlines and other escapecharacters

    #tokenize and lowercase
    tokens = tokenize(text, lentoken)#the 0 indicates that we remove empty tokens

    # importing LabMT for sentiment score
    labmt = pd.read_csv('/Users/kennethenevoldsen/Dropbox/Aarhus Universitet/Text Mining/Scripts/LabMT (sentiment analysis)/labmt_dict.csv',
                    sep = '\t', encoding = 'utf-8', index_col = 0) #Change filepath if you are using a different computer

    avg = labmt.happiness_average.mean() #this is done to averaging around zero instead of the 1-10 scale
    sent_dict = (labmt.happiness_average - avg).to_dict() #to.dict() er en pandas function - derfor antager den at

    #apply the LabMT sentiment score
    result = sum(sent_dict.get(token, 0.0) for token in tokens) #append to the sent_vect the sum of the sentiment scores for the each token in text

    if rel:
        result = result/len(tokens)
    #return result
    return result

#########################################

    #plotting functions from kristoffer's quick'n'dirty'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def plotvars(x,y = 0, sv = False, filename = 'qd_plot.png', ax1 = 'x', ax2 = 'f(x)'):
    """
    quick and dirty x and x-y plotting
    """
    fig, ax = plt.subplots()
    if y:
        ax.plot(x,y, color = 'k')
        ax.set_xlabel(ax1)
        ax.set_ylabel(ax2)
    else:
        ax.plot(x, color = 'k', linewidth = .5)
        ax.set_xlabel(ax1)
        ax.set_ylabel(ax2)
    if sv:
        plt.savefig(filename, dpi = 300)
    else:
        plt.show()
        plt.close()

def plotdist(x, sv = 0, filename = "dist.png"):
    """ histogram with normal fit """
    mu = np.mean(x)
    sigma =  np.std(x)
    n, bins, patches = plt.hist(x, 50, normed=1, facecolor='k', alpha=0.75)
    y = mlab.normpdf(bins, mu, sigma)# best normal fit
    ax = plt.plot(bins, y, 'r--', linewidth=1)
    plt.ylabel('Probability')
    plt.grid(True)
    if sv == 1:
        plt.savefig(filename, dpi = 300)
    else:
        plt.show()
        plt.close()
