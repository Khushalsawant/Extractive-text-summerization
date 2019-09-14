# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:07:09 2019

@author: khushal
"""

## https://www.dataquest.io/blog/web-scraping-tutorial-python/

'''
from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

# Create a placeholder for model
model = defaultdict(lambda: defaultdict(lambda: 0))

# Count frequency of co-occurance  
for sentence in reuters.sents():
    for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
        model[(w1, w2)][w3] += 1
 
# Let's transform the counts to probabilities
for w1_w2 in model:
    total_count = float(sum(model[w1_w2].values()))
    for w3 in model[w1_w2]:
        model[w1_w2][w3] /= total_count

import random

# starting words
text = ["today", "the"]
sentence_finished = False
 
while not sentence_finished:
  # select a random probability threshold  
  r = random.random()
  accumulator = .0

  for word in model[tuple(text[-2:])].keys():
      accumulator += model[tuple(text[-2:])][word]
      # select words that are above the probability threshold
      if accumulator >= r:
          text.append(word)
          break

  if text[-2:] == [None, None]:
      sentence_finished = True
 
print (' '.join([t for t in text if t]))
'''

import sys 
from nltk.corpus import brown
from nltk.corpus import reuters
import nltk
from nltk.corpus import PlaintextCorpusReader

def get_trigram_freq(tokens):
    tgs = list(nltk.trigrams(tokens))

    a,b,c = list(zip(*tgs))
    bgs = list(zip(a,b))
    return nltk.ConditionalFreqDist(list(zip(bgs, c)))

def get_bigram_freq(tokens):
    bgs = list(nltk.bigrams(tokens))

    return nltk.ConditionalFreqDist(bgs)

def appendwithcheck (preds, to_append):
    for pred in preds:
        if pred[0] == to_append[0]:
            return
    preds.append(to_append)

def incomplete_pred(words, n):
    all_succeeding = bgs_freq[(words[n-2])].most_common()
    print (all_succeeding, file=sys.stderr)
    preds = []
    number=0
    for pred in all_succeeding:
        if pred[0].startswith(words[n-1]):
            appendwithcheck(preds, pred)
            number+=1
        if number==3:
            return preds
    if len(preds)<3:
        med=[]
        for pred in all_succeeding:
            med.append((pred[0], nltk.edit_distance(pred[0],words[n-1], transpositions=True)))
        med.sort(key=lambda x:x[1])
        index=0
        while len(preds)<3:
            print (index, len(med))
            if index<len(med):
                if med[index][1]>0:
                    appendwithcheck(preds, med[index])
                index+=1
            if index>=len(preds):
                return preds

    return preds

new_corpus = PlaintextCorpusReader('./','.*')

#tokens = nltk.word_tokenize(raw)
tokens = brown.words() + new_corpus.words('my_corpus.txt')
#tokens = reuters.words()

#compute frequency distribution for all the bigrams and trigrams in the text
bgs_freq = get_bigram_freq(tokens)
tgs_freq = get_trigram_freq(tokens)

string = 'This is '

words=string.split()

n=len(words)
print (bgs_freq[(string)].most_common(5),file=sys.stderr)          
print (tgs_freq[(words[n-2],words[n-1])].most_common(5),file=sys.stderr)
pred_output  = incomplete_pred(words, n)
print("pred_output = ",pred_output)