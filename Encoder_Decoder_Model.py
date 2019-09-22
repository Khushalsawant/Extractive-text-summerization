# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:53:03 2019

@author: khushal
"""

import numpy as np
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense

from keras import backend as K 
K.clear_session() 

# generate a sequence of random integers
def generate_sequence(length, n_unique):
	return [randint(1, n_unique-1) for _ in range(length)]

# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
        X1, X2, y = list(), list(), list()
        for _ in range(n_samples):
                # generate source sequence
                source = generate_sequence(n_in, cardinality)
                # define padded target sequence
                target = source[:n_out]
                target.reverse()
                # create padded input target sequence
                target_in = [0] + target[:-1]
                # encode
                src_encoded = to_categorical([source], num_classes=cardinality)
                tar_encoded = to_categorical([target], num_classes=cardinality)
                tar2_encoded = to_categorical([target_in], num_classes=cardinality)
                # store
                X1.append(src_encoded)
                X2.append(tar2_encoded)
                y.append(tar_encoded)
        X1 = np.squeeze(array(X1), axis=1) 
        X2 = np.squeeze(array(X2), axis=1) 
        y = np.squeeze(array(y), axis=1) 
        return X1, X2, y

# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units):
    '''
    n_input: The cardinality of the input sequence, e.g. number of features, words, or characters for each time step.
    n_output: The cardinality of the output sequence, e.g. number of features, words, or characters for each time step.
    n_units: The number of cells to create in the encoder and decoder models, e.g. 128 or 256.
    '''
	# define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
	# define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	# define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
	# define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
	# return all models
    '''
    train: Model that can be trained given source, target, and shifted target sequences.
    inference_encoder: Encoder model used when making a prediction for a new source sequence.
    inference_decoder Decoder model use when making a prediction for a new source sequence.
    '''
    print("model.summary = ",model.summary())
    print("encoder_model.summary = ",encoder_model.summary())
    print("decoder_model.summary = ",decoder_model.summary())
    
    return model, encoder_model, decoder_model

# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    '''
    infenc: Encoder model used when making a prediction for a new source sequence.
    infdec: Decoder model use when making a prediction for a new source sequence.
    source:Encoded source sequence.
    n_steps: Number of time steps in the target sequence.
    cardinality: The cardinality of the output sequence, e.g. the number of features, words, or characters for each time step.
    '''

	# encode
    state = infenc.predict(source)
	# start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
	# collect predictions
    output = list()
    for t in range(n_steps):
		# predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
		# store prediction
        output.append(yhat[0,0,:])
		# update state
        state = [h, c]
		# update target sequence
        target_seq = yhat
    return array(output)

# decode a one hot encoded string
def one_hot_decode(encoded_seq):
	return [argmax(vector) for vector in encoded_seq]

# configure problem
n_features = 50 + 1
n_steps_in = 6
n_steps_out = 3
# define model
train, infenc, infdec = define_models(n_features, n_features, 128)
train.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 100000)
print(X1.shape,X2.shape,y.shape)
#print([X1, X2], y)
# train model
train.fit([X1, X2], y, epochs=1)
# evaluate LSTM
total, correct = 100, 0

for _ in range(total):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	if array_equal(one_hot_decode(y[0]), one_hot_decode(target)):
		correct += 1
        
print('Accuracy: %.2f%%' % (float(correct)/float(total)*100.0))

# spot check some examples
for _ in range(10):
	X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
	target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
	print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target)))