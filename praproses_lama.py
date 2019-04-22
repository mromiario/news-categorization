#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:08:39 2018

@author: mromiario
"""

from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


factori = StemmerFactory()
stemmer = factori.create_stemmer()
 
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


def praproses_data(input_path, stopword=stopword, stemmer=stemmer):
    df = pd.read_excel(input_path)
    data = df.iloc[:,2].values.tolist()
    bow = {}
    data_bersih = []
    for i, row in enumerate(data):
        lowcase_word = row.lower()       #lowcase data perbaris
        stopw = stopword.remove(lowcase_word)
        stemming = stemmer.stem(stopw)
        tokenizer = RegexpTokenizer(r'\w+')         #remove punctuatuion
        tokens = tokenizer.tokenize(stemming)       #Tokenisasi Kalimat
        kalimat = ''
        for kata in tokens:
            if kata not in bow :
                bow[kata] = 1
            else:
                bow[kata] += 1
            kalimat += kata + ' '
        print(kalimat)
        data_bersih.append(kalimat)
    a = np.array(data_bersih)
    np.save("bow.npy", bow)
    np.savetxt("data_bersih.csv", a, fmt='%s')
    return data_bersih


data_bersih = praproses_data('data10.xlsx')

def tf_idf(corpus):
    tf = TfidfVectorizer()
    X = tf.fit_transform(corpus).toarray()
    Y = tf.get_feature_names()
    return X,Y,tf

X,Y,tf = tf_idf(data_bersih)
