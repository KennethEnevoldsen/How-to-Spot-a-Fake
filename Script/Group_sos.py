#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 15:22:29 2017

@author: kennethenevoldsen

Text Mining - Group SOS
"""

    #import modules and setting wd
from __future__ import division
import os, re
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from nltk.tag import pos_tag
from gensim import corpora, models

import eli5

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegressionCV

wd = "/Users/kennethenevoldsen/Dropbox/Aarhus Universitet/Text Mining/Scripts/group_sos" #wd - absolute path
os.chdir(wd)

import textMiningModule as tm
import textminer as tm1 #a bit sloppy I know, should probably merge the two merge the two modules together...


#########################################


    #importing modified data
df = pd.read_csv('fake_or_real_news_cleaned_sent.csv', encoding = 'utf-8')
print df.label.value_counts() #balanced dataset (approx 3000 of each)
print df.loc[1]


"""
    #importing original data
df = pd.read_csv('fake_or_real_news.csv', encoding = 'utf-8')
print df.label.value_counts() #balanced dataset (approx 3000 of each)

    #removing bad rows
bad_rows = []
for row in range(len(df)):              #loop through all rows in the df
    if len(df.loc[row, 'text']) < 100:   #if the len of the text is > 100:
        bad_rows.append(row)            #save the row in the bad_row list

df = df.drop(df.index[bad_rows]) #remove all the bad rows
print len(bad_rows) #print the ammount of removed data
print df.label.value_counts() #still a balanced dataset

"""
"""
#########################################

    #ADDING TO THE DATAFRAME

###

    #making column with absolute sentiment scores

text_scored = []
for text in df['text']:
    sent_score = tm.labMT_sent(text)
    text_scored.append(sent_score)

df['abs_sent'] = text_scored #adding the scored text to the df



####

    #making column with relative sentiment scores

text_scored = []
for text in df['text']:
    sent_score = tm.labMT_sent(text, rel = True)
    text_scored.append(sent_score)

df['rel_sent'] = text_scored #adding the scored text to the df


###

    #Making column with text length

text_len = []
for text in df['text']:
    text_len.append(len(text))

df['text_len'] = text_len  #adding the length of the text to the df


###

    #Making column with title length

title_len = []
for text in df['title']:
    title_len.append(len(text))

df['title_len'] = title_len  #adding the length of the title to the df


###

    #Making column with cleaned text

text_clean = []
for text in df['text']:
    text = re.sub(r'[^a-zA-Z]', ' ', text) #replace everything that is not alfapethical with a space
    text = re.sub(r' +', ' ', text) #replace one or more whitespaces with a whitespace
    text = text.rstrip() #remove newlines and other escapecharacters
    text_clean.append(text)

df['text_clean'] = text_clean #adding the clean text to the df


###

    #Making column with cleaned titles

title_clean = []
for text in df['title']:
    text = re.sub(r'[^a-zA-Z]', ' ', text) #replace everything that is not alfapethical with a space
    text = re.sub(r' +', ' ', text) #replace one or more whitespaces with a whitespace
    text = text.rstrip() #remove newlines and other escapecharacters
    title_clean.append(text)

df['title_clean'] = title_clean #adding the clean title to the df


#saves the modifies df
df.to_csv('fake_or_real_news_cleaned_sent.csv', index = False, encoding = 'utf-8')#this saves the df to a csv file

"""



#########################################

    #SPLITTING THE DATASET
ratio = 0.8 #the ratio on which we split the data
mask = np.random.rand(len(df)) < ratio #creating random x random numbers from 0-1 where x is the len(df) - remove everything that is higher than ratio
data_train = df[mask]
data_test = df[~mask]

"""
print len(data_train)/len(df) #not exactly equal to ratio - that is due to the random number generator
"""

"""
#if you want to use the exact data_train and data_test as i uses in my analysis run these lines:
data_test = pd.read_csv('fake_or_real_news_test.csv', encoding = 'utf-8')
data_train = pd.read_csv('fake_or_real_news_train.csv', encoding = 'utf-8')

#to save the dataset run these:
data_test.to_csv('fake_or_real_news_test.csv', index = False, encoding = 'utf-8')
data_train.to_csv('fake_or_real_news_train.csv', index = False, encoding = 'utf-8')
"""

#########################################

df_topic = pd.read_csv('fake_or_real_news_topics10.csv', encoding = 'utf-8') #this is trained based on the 'fake_or_real_news_train.csv'
df_topic = pd.read_csv('fake_or_real_news_topics20.csv', encoding = 'utf-8')
df_topic = pd.read_csv('fake_or_real_news_topics15.csv', encoding = 'utf-8')

"""
    #MAKING A TOPIC MODEL DATAFRAME
wdf = data_train #defining a working df  - change this when we want to work with all of the texts
#insert articles into a list


texts_tokenized = []
for text in wdf['text_clean']:
    tokens = tm1.tokenize(text, length = 1, casefold = False) #casefold equal false because we want uppercase letters to categorize the text using pos_tag
    tagset = pos_tag(tokens, tagset = 'universal', lang = 'eng') #tag tokens with their category
    tokens = [tag[0] for tag in tagset if tag[1] in ['NOUN']] #only retain nouns
    tokens = [token.lower() for token in tokens] #lowercase the tokens
    texts_tokenized.append(tokens)

print type(texts_tokenized[0][0])  #the word in the text
print type(texts_tokenized[0])  #list of words in text
print type(texts_tokenized) #list of texts
#So it is a string within a list within a list (the first list is the text, the second list the nouns in the text and the string is the noun)

    #making a stopwordlist
sw = tm1.gen_ls_stoplist(texts_tokenized, 100)
print sw #this stopword might say some general things about the period of the articles rather than something about the topics
#for now let's just not use it

#applying stopword list to all texts
nosw_texts = []
for text in texts_tokenized:
    nosw_text = []
    nosw_text =[token for token in text if token not in sw]
    nosw_texts.append(nosw_text)

texts_tokenized = nosw_texts#overwrite with text without the stopwords

dictionary = corpora.Dictionary(texts_tokenized)

    #bag-of-words representation of the texts
texts_bow = [dictionary.doc2bow(text) for text in texts_tokenized] #bow = bag of words

    #training the topic model
k = 10 #number of topics
mdl = models.LdaModel(texts_bow, id2word = dictionary,
                      num_topics = k, random_state = 12345) #does LdaModel stands for linear discriminant model?

for i in range(k): #here we will the the first ten words in each topic
    print 'topic', i
    print [t[0] for t in mdl.show_topic(i, 20)]
    print '----'

#redefing wdf to become the whole df (because we need the topic result for the whole dataframe)
wdf = df
texts_tokenized = []
for text in wdf['text_clean']:
    tokens = tm1.tokenize(text, length = 1, casefold = False) #casefold equal false because we want uppercase letters to categorize the text using pos_tag
    tagset = pos_tag(tokens, tagset = 'universal', lang = 'eng') #tag tokens with their category
    tokens = [tag[0] for tag in tagset if tag[1] in ['NOUN']] #only retain nouns
    tokens = [token.lower() for token in tokens] #lowercase the tokens
    texts_tokenized.append(tokens)
texts_bow = [dictionary.doc2bow(text) for text in texts_tokenized] #bow = bag of words

    #adding the topics to a dataframe
def get_theta(doc_bow, mdl):
    tmp = mdl.get_document_topics(doc_bow, minimum_probability=0)
    return [p[1] for p in tmp]

df_topic = pd.DataFrame() #making empty dataframe

for topicnr in range(k):
    topic_name = 'topic %d' %topicnr
    topic_score = []
    for text in range(len(wdf)):
        topic_score.append(get_theta(texts_bow[text], mdl)[topicnr])
    topic_name = 'topic %d' %topicnr
    df_topic[topic_name] = topic_score



print mdl[texts_bow[1]] #show the topic of the text
print df_topic.head()
print len(df_topic)

df_topic.to_csv('fake_or_real_news_topics15.csv', index = False, encoding = 'utf-8')
df_topic.to_csv('fake_or_real_news_topics20.csv', index = False, encoding = 'utf-8')
df_topic.to_csv('fake_or_real_news_topics10.csv', index = False, encoding = 'utf-8')
"""
#########################################

    #DEFINING TRAINING FEATURES

###first classifier (clf1 - sentiment)

    #training representation
train_feat_1 = data_train.as_matrix(columns=['abs_sent', 'rel_sent']) #making training feature matrix
train_y = data_train['label'].values #classes

    #test representation
test_feat_1 = data_test.as_matrix(columns=['abs_sent', 'rel_sent']) #making training feature matrix
test_y = data_test['label'].values #classes

###second classifier (clf2 - word freq)

vecspc = CountVectorizer() #instantiate vectorizer which count frequency
    #training representation
train_x = data_train['text_clean'].values
train_feat_2 = vecspc.fit_transform(train_x)
    #test representation
test_x = data_test['text_clean'].values
test_feat_2 = vecspc.transform(test_x)

###Third classifier (clf3 - Topic)

"""
#making data_topic_ have the same mask data_train () - same for data_topic_test - it should be done like this as apposed to the mask if you want the same result (as the mask is randomized each time you run the script)
df_topic['ID'] = df['ID']
templist_train = [] # for training set
for i in df_topic['ID']:
    if i in data_train['ID'].tolist():
        templist_train.append(i)
data_topic_train = df_topic.loc[df_topic['ID'].isin(templist_train)]
len(data_topic_train)
del data_topic_train['ID']

templist_test = [] #for test set
for i in df_topic['ID']:
    if i in data_test['ID'].tolist():
        templist_test.append(i)
data_topic_test = df_topic.loc[df_topic['ID'].isin(templist_test)]
len(data_test)
len(data_topic_test)
del data_topic_test['ID']
del df_topic['ID']
"""


#this can also be done like this
data_topic_train = df_topic[mask]
data_topic_test = df_topic[~mask]


train_feat_3 = data_topic_train.as_matrix() #making training feature matrix
test_feat_3 = data_topic_test.as_matrix() #making training feature matrix

###Fourth classifier (clf4 - Title)
vecspc2 = CountVectorizer()
train_x = data_train['title'].values
train_feat_4 = vecspc2.fit_transform(train_x)
test_x = data_test['title'].values
test_feat_4 = vecspc2.transform(test_x)

###Fifth Classifier (clf5 - tf-idf)
vecspc3 = TfidfVectorizer() #instantiate vectorizer which use TF-IDF (term frequency and inverse frequency)
    #training representation
train_x = data_train['text_clean'].values
train_feat_5 = vecspc3.fit_transform(train_x)
    #test representation
test_x = data_test['text_clean'].values
test_feat_5 = vecspc3.transform(test_x)


#########################################

    #TRAINING
    #training a logistic regression

#training classifier 1
clf1 = LogisticRegressionCV() #intantiate a classifier
clf1.fit(train_feat_1, train_y)#fit our training feature to our classes

#training classifier 2
clf2 = LogisticRegressionCV() #intantiate a classifier
clf2.fit(train_feat_2, train_y)#fit our training feature to our classes

#training classifier 3
clf3 = LogisticRegressionCV() #intantiate a classifier
clf3.fit(train_feat_3, train_y)#fit our training feature to our classes

#training classifier 4
clf4 = LogisticRegressionCV() #intantiate a classifier
clf4.fit(train_feat_4, train_y)#fit our training feature to our classes

#training classifier 5
clf5 = LogisticRegressionCV() #intantiate a classifier
clf5.fit(train_feat_5, train_y)#fit our training feature to our classes

"""
###
    #Saving  classifiers
with open('classlogreg_news_1.pcl', 'wb') as f:
    pickle.dump(clf1, f)
with open('classlogreg_news_2.pcl', 'wb') as f:
    pickle.dump(clf2, f)
with open('classlogreg_news_3_10.pcl', 'wb') as f: #this is for 10 topics
    pickle.dump(clf3, f)
with open('classlogreg_news_3_15.pcl', 'wb') as f: #this is for 15 topics
    pickle.dump(clf3, f)
with open('classlogreg_news_3_20.pcl', 'wb') as f: #this is for 20 topics
    pickle.dump(clf3, f)
with open('classlogreg_news_4.pcl', 'wb') as f:
    pickle.dump(clf4, f)
with open('classlogreg_news_5.pcl', 'wb') as f:
    pickle.dump(clf5, f)
"""
"""
###
    #import classifiers
clf1 = pickle.load(open('classlogreg_news_1.pcl', 'r'))
clf2 = pickle.load(open('classlogreg_news_2.pcl', 'r'))
clf3 = pickle.load(open('classlogreg_news_3_10.pcl', 'r'))
clf3 = pickle.load(open('classlogreg_news_3_15.pcl', 'r'))
clf3 = pickle.load(open('classlogreg_news_3_20.pcl', 'r'))
clf4 = pickle.load(open('classlogreg_news_4.pcl', 'r'))
clf5 = pickle.load(open('classlogreg_news_5.pcl', 'r'))
"""

#########################################

    #VALIDATION

    #clf1
pred1 = clf1.predict(test_feat_1)
#testing performance metrics using a confusion matrix - clf1
conf_mat_1 = metrics.confusion_matrix(test_y, pred1)
print '\n', conf_mat_1

perf_acc_1 = metrics.accuracy_score(test_y, pred1)
print perf_acc_1

print '\n', metrics.classification_report(test_y, pred1)


    #clf2
pred2 = clf2.predict(test_feat_2)
#testing performance metrics using a confusion matrix - clf2
conf_mat_2 = metrics.confusion_matrix(test_y, pred2)
print '\n', conf_mat_2

perf_acc_2 = metrics.accuracy_score(test_y, pred2)
print perf_acc_2

print '\n', metrics.classification_report(test_y, pred2)


    #clf3
pred3 = clf3.predict(test_feat_3)
#testing performance metrics using a confusion matrix - clf3
conf_mat_3 = metrics.confusion_matrix(test_y, pred3)
print '\n', conf_mat_3

perf_acc_3 = metrics.accuracy_score(test_y, pred3)
print perf_acc_3

print '\n', metrics.classification_report(test_y, pred3)


    #clf4
pred4 = clf4.predict(test_feat_4)
#testing performance metrics using a confusion matrix - clf4
conf_mat_4 = metrics.confusion_matrix(test_y, pred4)
print '\n', conf_mat_4

perf_acc_4 = metrics.accuracy_score(test_y, pred4)
print perf_acc_4


print '\n', metrics.classification_report(test_y, pred4)


    #clf5
pred5 = clf5.predict(test_feat_5)
#testing performance metrics using a confusion matrix - clf4
conf_mat_5 = metrics.confusion_matrix(test_y, pred5)
print '\n', conf_mat_5

perf_acc_5 = metrics.accuracy_score(test_y, pred5)
print perf_acc_5


print '\n', metrics.classification_report(test_y, pred5)




#########################################

    #EXPLORE CLASSIFIERS

    #clf2
#which articles didn't it predict correctly
pred2_list = [tup[0:4] for tup in pred2] #converting np array to list
wrong_pred = []
templist = data_test['label'].tolist()
for i in range(len(pred2_list)):
    if pred2_list[i] !=  templist[i]:
        wrong_pred.append(i)

n = 8
wrong_pred[n]
print data_test.iloc[wrong_pred[n], 2] # print the text the article
print data_test.iloc[1, 0]

#making a df containing only wrongly classified real news
templist_real = [] #for test set
for i in wrong_pred:
    if data_test.iloc[i, 3] == 'REAL':
        templist_real.append(data_test.iloc[i, 0])
df_wrongpred_r = data_test.loc[data_test['ID'].isin(templist_real)]
df_wrongpred_r.head(61)

bad_rows = []
for row in range(len(df)):              #loop through all rows in the df
    if len(df.loc[row, 'text']) < 100:   #if the len of the text is > 100:
        bad_rows.append(row)            #save the row in the bad_row list

df = df.drop(df.index[bad_rows]) #remove all the bad rows


#show most prominent words
eli5.show_weights(clf2, vec = vecspc, top=30)

#highlight words in text
i = 73
i = 201
text = df['text_clean'][i]
print df['label'][i]
eli5.show_prediction(clf2, text, vec=vecspc)



    #clf4
eli5.show_weights(clf4, vec = vecspc2, top=30)




eli5.show_prediction(clf4, df['title_clean'][99], vec=vecspc2) #99 is pretty good

    #clf5
eli5.show_weights(clf5, vec = vecspc3, top=30)




eli5.show_prediction(clf4, df['title_clean'][99], vec=vecspc2) #99 is pretty good
eli5.show_prediction(clf4, df['title'][176], vec=vecspc2)
eli5.show_prediction(clf4, df['title_clean'][5988], vec=vecspc2)
eli5.show_prediction(clf4, df['title_clean'][6000], vec=vecspc2)


###

    #Most informative features

def show_most_informative_features(clf, featurenames,  n=20):
    feature_names = featurenames #det er antallet af topics
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)



show_most_informative_features(clf3, range(k), k) #logistical regression (coefficients) -- k is the number of topics

show_most_informative_features(clf1, ['abs_sent', 'rel_sent'], 2)

###############################

    #VISUALIZATION

    #making a heatmap

#making the wanted dataframe for heatmap (can only contain raw numbers)
df_topic['label'] = df['label']
df_topic = df_topic.sort_values('label')

df_topic.head(1)

del df_topic['label']
heat_matrix = df_topic.as_matrix() #make it into a matrix (again only numbers)

#this is not always necesarry - however if the next line does not show the heatmap run this (does not work in spyder)
%matplotlib inline

ax = sns.heatmap(heat_matrix, yticklabels=False) #make the heats






    #making a wordcloud
import random
from wordcloud import WordCloud



#########################################

    #RANDOM CALCULATIONS

#absolute score mean calculations
df['abs_sent'].mean() #overall mean
df['abs_sent'].loc[df['label'] == 'FAKE'].mean() #'fake' mean
df['abs_sent'].loc[df['label'] == 'REAL'].mean() #'real' mean
#relative score mean calculations
df['rel_sent'].mean() #overall mean
df['rel_sent'].loc[df['label'] == 'FAKE'].mean() #'fake' mean
df['rel_sent'].loc[df['label'] == 'REAL'].mean() #'real' mean

#mean length of titles and body text
df['text_len'].mean() / df['title_len'].mean()

df.tail(220)
