import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano


vocabulary_size = 11
unknown_token = "UNK"

# sentence_end_token = "SENTENCE_END"


# Outer SGD Loop
# - model: The RNN model instance
# - X_train: The training data set
# - y_train: The training data labels
# - learning_rate: Initial learning rate for SGD
# - nepoch: Number of times to iterate through the complete dataset
# - evaluate_loss_after: Evaluate the loss after this many epochs
def train_with_sgd(model, X_train, y_train, learning_rate=0.05, nepoch=5000, evaluate_loss_after=5):
  # We keep track of the losses so we can plot them later
  losses = []
  num_examples_seen = 0
  for epoch in range(nepoch):
    # Optionally evaluate the loss
    if (epoch % evaluate_loss_after == 0):
      loss = model.calculate_loss(X_train, y_train)
      losses.append((num_examples_seen, loss))
      time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      # print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
      # Adjust the learning rate if loss increases
      if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
          learning_rate = learning_rate * 0.5  
          print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
          print "Setting learning rate to %f" % learning_rate
          if learning_rate < 0.000001:break
      sys.stdout.flush()
    # For each training example...
    for i in range(len(y_train)):
        # One SGD step
        print model.sgd_step(X_train[i], y_train[i], learning_rate)
        # model.sgd_step(X_train[i], y_train[i], learning_rate)
        num_examples_seen += 1

 
# Read the data and append SENTENCE_START and SENTENCE_END tokens
print "Reading CSV file..."
# with open('data/reddit-comments-2015-08.csv', 'rb') as f:
#   reader = csv.reader(f, skipinitialspace=True)
#   reader.next()
#   # Split full comments into sentences
#   # sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
#   # Append SENTENCE_START and SENTENCE_END
#   sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
# print "Parsed %d sentences." % (len(sentences))
with open('data/train.txt', 'rb') as f:
  sentences = f.readlines()
  # sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
  # sentences = ["%s %s" % (x, sentence_end_token) for x in sentences]
print "Parsed %d sentences." % (len(sentences))   
# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
 
# Count the word frequencies
word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
print "Found %d unique words tokens." % len(word_freq.items())
 
# Get the most common words and build index_to_word and word_to_index vectors
vocab = word_freq.most_common(vocabulary_size)
# index_to_word = [x[0] for x in vocab]
# index_to_word.append(unknown_token)
index_to_word = []
for i in range(0,10):
  index_to_word.append(str(i))
# index_to_word.append(sentence_start_token) #10
# index_to_word.append(sentence_end_token) #11
index_to_word.append(unknown_token) #12
word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

 
print "Using vocabulary size %d." % vocabulary_size
print "The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1])
 
# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
  tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
 
print "\nExample sentence: '%s'" % sentences[0]
print "\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0]


# Create the training data
X_train = np.asarray([[np.int32(word_to_index[w]) for w in sent] for sent in tokenized_sentences[:-1]])
Y_train = np.asarray([[np.int32(word_to_index[w]) for w in sent] for sent in tokenized_sentences[1:]])

print X_train, type(X_train)
print Y_train, type(Y_train)

np.random.seed(10)
model = RNNTheano(vocabulary_size)
# model = RNNNumpy(vocabulary_size)
# o, s = model.forward_propagation(X_train[1])
# print o.shape
# print o
l = [8,9,0,1,2,3,4,5,6,7]
x = np.asarray([np.int32(a) for a in l])
# x = np.asarray([np.int32(a) for a in range(0,10)])
# print x, type(x)
print "input", x
# x[0] = 10
# print x, type(x)
# o = model.forward_propagation(x)
# print "o.shape",(o).shape, o 

predictions = model.predict(x)
# print predictions.shape
print "befor trained", predictions

# %timeit model.sgd_step(X_train[10], y_train[10], 0.005)
# train_with_sgd(model, X_train, Y_train, nepoch=10)
train_with_sgd(model, X_train, Y_train, nepoch=5)
predictions = model.predict(x)
print "after trained", predictions