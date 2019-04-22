# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data_bersih.csv', header=None).values.tolist()


#print(df)


#tfidf = TfidfVectorizer(
#    analyzer='word',
#    tokenizer=df,
#    preprocessor=df,
#    token_pattern=None)

def tf_idf(corpus):
    tf = TfidfVectorizer()
    X = tf.fit_transform(corpus).toarray()
    Y = tf.get_feature_names()
    return X,Y,tf

X,Y,tf = tf_idf(data_bersih)