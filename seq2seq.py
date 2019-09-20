# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:15:58 2019

@author: KS5046082
"""


# https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/

# https://machinelearningmastery.com/timedistributed-layer-for-long-short-term-memory-networks-in-python/
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential,Input,Model
from keras.layers import Dense,Dropout,LSTM,concatenate,Embedding
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot

import re
import math   

# define the document
path_of_file =  "D:/extracted_data_file.txt"
txt_file = open(path_of_file,"r")
text = txt_file.readlines() 
txt_file.close() 

print(type(text))
text_str = "".join(text)
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text_str))
vocab_size = len(words)
print("vocab_size = ",vocab_size)

# using regex (findall()) 
# to count words in string 
res = len(re.findall(r'\w+', text_str)) 
# printing result 
print ("The number of words in string are : " +  str(res)) 

# integer encode the document
result = one_hot(text_str, round(vocab_size*1.3))
#print(result)

tokenizer = Tokenizer(num_words = 3000  )

src_txt_length = len(text_str)
sum_txt_length = math.ceil(src_txt_length*0.3)

print("sum_txt_length = ",sum_txt_length)
print("src_txt_length = " ,src_txt_length)
# source text input model
inputs1 = Input(shape=(src_txt_length,))
am1 = Embedding(vocab_size, 128)(inputs1)
am2 = LSTM(128)(am1)
# summary input model
inputs2 = Input(shape=(sum_txt_length,))
sm1 = Embedding(vocab_size, 128)(inputs2)
sm2 = LSTM(128)(sm1)
# decoder output model
decoder1 = concatenate([am2, sm2])
outputs = Dense(vocab_size, activation='softmax')(decoder1)
# tie it together [article, summary] [word]
model = Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam')
#model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())