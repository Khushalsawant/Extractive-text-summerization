# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 13:53:03 2019

@author: khushal
"""

from tkinter import *
from tkinter.messagebox import *
import tkinter as tk
from Extractive_summerization import summarized_data_from_wiki


def printtext():
    global e1
    string = e1.get() 
    print(string)
    summary,no_of_sentences_in_summary = summarized_data_from_wiki(string)
    print(summary)
    t1.config(text=summary)

window = Tk()

window.title('Topic to get Exractive summarized data')
window.minsize(width=300, height=150)

#size of the window
#window.geometry("400x300")

l1 = Label(window, text="Input for Exractive summarized data")
l1.pack(side='left')
e1 = Entry(window)
e1.focus_set()

e1.pack(side='right')

    # if you want the button to disappear:
    # button.destroy() or button.pack_forget()
t1 = Label(window)


b1 = Button(window,text='Okay',command=printtext)

# allowing the widget to take the full space of the window window
b1.pack(side='bottom')

window.mainloop()