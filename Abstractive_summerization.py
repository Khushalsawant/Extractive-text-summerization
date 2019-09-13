# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 07:30:09 2019

@author: khushal
"""

import time
# into hours, minutes and seconds 
import datetime 

start_time = time.time()

import requests
from bs4 import BeautifulSoup

import wikipedia
from langdetect import detect

import nltk
import os
import re
import math
import operator
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
Stopwords = set(stopwords.words('english'))
wordlemmatizer = WordNetLemmatizer()

def convert_sec(n): 
    return str(datetime.timedelta(seconds = n))


def lemmatize_words(words):
    lemmatized_words = []
    for word in words:
       lemmatized_words.append(wordlemmatizer.lemmatize(word))
    return lemmatized_words
def stem_words(words):
    stemmed_words = []
    for word in words:
       stemmed_words.append(stemmer.stem(word))
    return stemmed_words
def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex,'',text)
    return text
def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
       if word not in words_unique:
           words_unique.append(word)
    for word in words_unique:
       dict_freq[word] = words.count(word)
    return dict_freq
def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word,tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
             pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb


def summarized_data_from_wiki(input_str):
    extracted_str = "/n"
    #print(wikipedia.summary("ETF"))
    ETF = wikipedia.page(input_str)
    print(ETF.url)
    extracted_list = []
    page = requests.get(ETF.url)
    if page.status_code == 200:
        
        spl_chr_list = []
        extracted_str_list = []
        for i in range(101):
            spl_chr = "[" + str(i) + "]"
            spl_chr_list.append(spl_chr)
        
        #print(page.content)
        
        soup = BeautifulSoup(page.content, 'html.parser')
        #print(soup.prettify())
        soup.find_all('p')
        for i in range(len(soup.find_all('p'))):
            extracted_str = extracted_str +soup.find_all('p')[i].get_text()
        
        for char in spl_chr_list:
            if extracted_str.find(char) != -1:
                extracted_str = extracted_str.replace(str(char),"")
                extracted_str_list.append(extracted_str)
    
    
    text = extracted_str_list[-1]
    
    lang = detect(text) # lang = 'en' for an English email
    print("Extracted data in present in", lang)
    tokenized_sentence = sent_tokenize(text)
    #print(" Length of tokenized_sentence",len(tokenized_sentence))
    
    text = remove_special_characters(str(text))
    text = re.sub(r'\d+', '', text)
    tokenized_words_with_stopwords = word_tokenize(text)
    tokenized_words = [word for word in tokenized_words_with_stopwords if word not in Stopwords]
    tokenized_words = [word for word in tokenized_words if len(word) > 1]
    tokenized_words = [word.lower() for word in tokenized_words]
    tokenized_words = lemmatize_words(tokenized_words)
    word_freq = freq(tokenized_words)
        
    input_user = 30 #int(input('Percentage of information to retain(in percent):'))
    print(" Percentage of information to retain(in percent): ",input_user)
    no_of_sentences = int((input_user * len(tokenized_sentence))/100)
    print(" no_of_sentences In Extracted data from wiki = ",no_of_sentences)    
    return text,no_of_sentences

def write_extracted_data_in_file(extracted_data):
    file = open("extracted_data_file.txt", "w")
    file.write(extracted_data) 
    file.close()

if __name__ == "__main__": 
    input_str = "ETF"# str(input('Enter the text/string for which yo want to get summarizd data := '))    
    text,no_of_sentences_in_summary = summarized_data_from_wiki(input_str)
    print("text: \n",text)
    print(" no_of_sentences In text data from wiki = ",no_of_sentences_in_summary)
    write_extracted_data_in_file(text)
    
    n =  time.time() - start_time
        
    print("---Execution Time ---",convert_sec(n))