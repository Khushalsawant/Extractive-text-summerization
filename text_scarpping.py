# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:07:09 2019

@author: khushal
"""

## https://www.dataquest.io/blog/web-scraping-tutorial-python/


import time
# into hours, minutes and seconds 
import datetime 


start_time = time.time()

'''
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

extracted_str = "/n"
page = requests.get("https://www.investopedia.com/terms/e/etf.asp")
if page.status_code == 200:
    #print(page.content)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup.prettify())
    soup.find_all('p')
    for i in range(len(soup.find_all('p'))):
        extracted_str = extracted_str + soup.find_all('p')[i].get_text()
''' 
import wikipedia
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize

extracted_str = "/n"
extracted_str_list = []
#print(wikipedia.summary("ETF"))
ETF = wikipedia.page("ETF")
print(ETF.url)
extracted_list = []
page = requests.get(ETF.url)

spl_chr_list = []
for i in range(101):
    spl_chr = "[" + str(i) + "]"
    spl_chr_list.append(spl_chr)
#print(spl_chr_list)    
if page.status_code == 200:
    #print(page.content)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup.prettify())
    soup.find_all('p')
    for i in range(len(soup.find_all('p'))):
        extracted_str = extracted_str + soup.find_all('p')[i].get_text()
    #print(len(extracted_str),extracted_str)
    
    for char in spl_chr_list:
        #print("char = ",char)
        if extracted_str.find(char) != -1:
            print("char = ",char)
            extracted_str = extracted_str.replace(str(char),"")
            extracted_str_list.append(extracted_str)
            #print(extracted_str.replace(str(char)," "))
            #break
        #print(extracted_str)

print(extracted_str_list[-1])
#print(ETF.content.split("."))
#print(type(ETF.content))
