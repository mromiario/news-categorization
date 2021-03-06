#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 15:08:39 2018

@author: 
    1301154199 Ilham Kurnia S
    1301154311 M Romi Ario
    1301154381 Gugun Mediamer
"""

from nltk.tokenize import RegexpTokenizer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import operator
import re


factori = StemmerFactory()
stemmer = factori.create_stemmer()
 
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()


def praproses_data(input_path, stopword=stopword, stemmer=stemmer):
    #I.S : Data hasil crawling
    #F.S : Data bersih yang sudah melalui preprocessing
    df = pd.read_excel(input_path)
    data = df.iloc[:,2].values.tolist()
    bow = {}
    data_bersih = []
    
    #stopword tambahan berdasarkan karakteristik data
    stopword_tambahan = ['all','artikel','bagi','berita','com','lain','okezone',
                         'reserved','rights','ini','itu','kait','jadi','kata'
                         ,'sebut','jakarta','laku','lebih','akan','ada','indonesia',
                         'dapat','yang','banyak','lalu','satu','tahun','tak',
                         'dunia','juga','baru','buah','lama','nama','beberapa',
                         'baik','guna','rupa','besar','akhir','belum','buat','diri','usaha','bisa']
    
    for i, row in enumerate(data):
        lowcase_word = row.lower()       #lowcase data perbaris
        stopw = stopword.remove(lowcase_word) #menghilangkan stopword Bahasa Indonesia
        stopw = re.sub('[^a-zA-Z!"\',]', ' ', str(stopw))#menghilangkan tanda baca dan angka
        stemming = stemmer.stem(stopw) #merubah ke bentuk kata dasar
        tokenizer = RegexpTokenizer(r'\w+')         #remove punctuatuion
        tokens = tokenizer.tokenize(stemming)       #Tokenisasi Kalimat
        
        #menghilangkan stopword tambahab
        tokens = [word for word in tokens if word not in stopword_tambahan]

        kalimat = ''
        for kata in tokens:     #menyimpan daftar kata
            if kata not in bow :
                bow[kata] = 1   
            else:
                bow[kata] += 1
            kalimat += kata + ' ' #menyimpan data yang sudah dilakukan preprocessing
        print(kalimat)
        data_bersih.append(kalimat) 
    #a = np.array(data_bersih)
    #np.save("bow.npy", bow)
    #np.savetxt("data_bersih.csv", a, fmt='%s')
    return data_bersih

def read_label(input_path):
    #I.S : Mendapat input file excel
    #F.S : Mengembalikan list berisi label setiap data
    #NOTE : KELAS 1 : EKONOMI, KELAS 2 : OLAHRAGA, KELAS 3: TEKNOLOGI, KELAS 4: SELEBRITI
    df = pd.read_excel(input_path)
    data = df.iloc[:,4].values.tolist()
    return data

def tf_idf(corpus):
    #I.S : Data bersih hasil preprocessing
    #F.S : Matriks TD-IDF, list fitur
    tf = TfidfVectorizer()
    X = tf.fit_transform(corpus).toarray()
    Y = tf.get_feature_names()
    return X,Y,tf

def SortedFrequentWord(label_kelas) :
    #I.S : Kelas yang ingin diurutkan frekuensi kemuculan katanya {1,2,3,4}
    #F.S : Menampilkan 13 kata paling banyak dari setiap kelas, menyimpan seluruh kata terurut pada list
    kelas = {}
    for i in range(len(data)) :
        for j in range(len(data[i])) :
            if (data[i][j] != 0 and label[i] == label_kelas): #Jika nilai fitur tersebut tidak nol dan kelasnya sesuai
                if fitur[j] in kelas :
                    kelas[fitur[j]] += 1 #menghitung DF pada kelas tertentu
                else :
                    kelas[fitur[j]] = 1
    kelas = sorted(kelas.items(), key=operator.itemgetter(1), reverse = True) #Mengurutkan dari nilai yang paling besar
    print('Kata dengan frekuensi paling banyak pada kelas ',label_kelas,' :') #Menampilkan 13 kata dengan kemunculan paling banyak
    print('')
    for i in range(13) :
        print(kelas[i][0])
    print('')
    return kelas

def CountDictionary(dict_word) :
    #I.S : Kamus tambahan dengan list kata yang sudah terdefinisi
    #F.S : List berisi kemunculan kata dalam biner pada seluruh dokumen
    class_binary = []
    for i in range(len(data)) :
        count = 0 #initial state
        for j in range(len(data[i])) :
            if(data[i][j] !=0 and (fitur[j] in dict_word)) : #Jika data tersebut ada di suatu dokumen dan terdapat juga di kamus yg didefinisikan
                count = 1
        class_binary.append(count)
    return class_binary

#MAIN PROGRAM 
    
data_bersih = praproses_data('data100.xlsx') #Preprocessing
data,fitur,tf = tf_idf(data_bersih) #TF IDF Weighting
label = read_label('data100.xlsx') #Collect label

kls1= SortedFrequentWord(1) #Sorting kata dg frekuensi terbanyak pada kelas Ekonomi
kls2 = SortedFrequentWord(2)#Sorting kata dg frekuensi terbanyak pada kelas Olahraga
kls3  = SortedFrequentWord(3)#Sorting kata dg frekuensi terbanyak pada kelas Teknologi
kls4  = SortedFrequentWord(4)#Sorting kata dg frekuensi terbanyak pada kelas Selebriti
 
#Kamus tambahan pada kelas Ekonomi   
word_class1 = ['menteri','kerja','ujar','jelas','pihak','bangun','perintah',
              'jalan','kalau','negara','utama','seluruh','hubung']

class1 = CountDictionary(word_class1) #Menghitung kemunculan kata pada tiap dokumen sesuai kamus tambahan

#Pembagian Data
data = np.column_stack((data,class1)) 
data_train = data[len(data)//2:] #100 Data Training
data_test = data[:len(data)//2] #1100 Data Testing


label_train = label[len(data)//2:]
label_test = label[:len(data)//2]

#Proses Training
clf = svm.SVC(kernel='linear')
clf.fit(data_train,label_train)


#Proses Prediksi
hasil_prediksi = clf.predict(data_test)
print('Akurasi :',clf.score(data_test,label_test)*100)
    





