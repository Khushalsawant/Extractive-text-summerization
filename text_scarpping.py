# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 07:07:09 2019

@author: khushal
"""

## https://www.dataquest.io/blog/web-scraping-tutorial-python/

import requests
from bs4 import BeautifulSoup

extracted_list = []
page = requests.get("https://www.investopedia.com/terms/e/etf.asp")
if page.status_code == 200:
    #print(page.content)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup.prettify())
    soup.find_all('p')
    for i in range(len(soup.find_all('p'))):
        extracted_list.append(soup.find_all('p')[i].get_text())
    print(len(extracted_list))
    print(extracted_list)
 
import wikipedia
import requests
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize

#print(wikipedia.summary("ETF"))
ETF = wikipedia.page("ETF")
print(ETF.url)
extracted_list = []
page = requests.get(ETF.url)
if page.status_code == 200:
    #print(page.content)
    soup = BeautifulSoup(page.content, 'html.parser')
    #print(soup.prettify())
    soup.find_all('p')
    for i in range(len(soup.find_all('p'))):
        a = sent_tokenize(soup.find_all('p')[i].get_text())
        print(a)
    print(extracted_list)
    print(len(extracted_list))
        
#print(ETF.content.split("."))
#print(type(ETF.content))
