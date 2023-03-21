
import pandas as pd
from collections import Counter
import nltk
import string
import re
from nltk.tokenize import (word_tokenize,
                           sent_tokenize,
                           TreebankWordTokenizer,
                           wordpunct_tokenize,
                           TweetTokenizer,
                           MWETokenizer
                          )
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

import math

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go

init_notebook_mode(connected=True)
from nltk.tokenize import sent_tokenize, word_tokenize

stop_words = stopwords.words()
from nltk.corpus import stopwords
stoplist = stopwords.words('english') + ['though']

rawdf = pd.read_csv (r"C:\Users\shanais.yuen\OneDrive - Seagroup\LiveSTORM\combined_livestorm.csv")
rawdf['Session date'] = pd.to_datetime(rawdf['Session date'])

rawdf["qns23"] = (rawdf[rawdf['Session date'].dt.year == 2023])['Question']
rawdf["qns22"] = (rawdf[rawdf['Session date'].dt.year == 2022])['Question']
rawdf["qns21"] = (rawdf[rawdf['Session date'].dt.year == 2021])['Question']


df23 = rawdf.filter(['Session date', 'qns23' ], axis=1)
df22 = rawdf.filter(['Session date', 'qns22' ], axis=1)
df21 = rawdf.filter(['Session date', 'qns21' ], axis=1)


"""""""""""""""""""""""""""
2023
"""""""""""""""""""""""""""

from textblob import TextBlob
df23['polarity'] = df23['qns23'].dropna().apply(lambda x: TextBlob(x).polarity)
df23['subjective'] = df23['qns23'].dropna().apply(lambda x: TextBlob(x).subjectivity)

from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) # bigram, trigram
#“convert a collection of text documents to a matrix of token counts

# matrix of ngrams
ngrams = c_vec.fit_transform(df23["qns23"].dropna())
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram23 = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
df_ngram23['polarity'] = df_ngram23['bigram/trigram'].apply(lambda x: TextBlob(x).polarity)
df_ngram23['subjective'] = df_ngram23['bigram/trigram'].apply(lambda x: TextBlob(x).subjectivity)


df_ngram23

"""""""""""""""""""""""""""
2022
"""""""""""""""""""""""""""


df22['polarity'] = df22['qns22'].dropna().apply(lambda x: TextBlob(x).polarity)
df22['subjective'] = df22['qns22'].dropna().apply(lambda x: TextBlob(x).subjectivity)

from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) # bigram, trigram
#“convert a collection of text documents to a matrix of token counts

# matrix of ngrams
ngrams = c_vec.fit_transform(df22["qns22"].dropna())
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram22 = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
df_ngram22['polarity'] = df_ngram22['bigram/trigram'].apply(lambda x: TextBlob(x).polarity)
df_ngram22['subjective'] = df_ngram22['bigram/trigram'].apply(lambda x: TextBlob(x).subjectivity)


df_ngram22



"""""""""""""""""""""""""""
2021
"""""""""""""""""""""""""""


df21['polarity'] = df21['qns21'].dropna().apply(lambda x: TextBlob(x).polarity)
df21['subjective'] = df21['qns21'].dropna().apply(lambda x: TextBlob(x).subjectivity)

from sklearn.feature_extraction.text import CountVectorizer
c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3)) # bigram, trigram
#“convert a collection of text documents to a matrix of token counts

# matrix of ngrams
ngrams = c_vec.fit_transform(df21["qns21"].dropna())
# count frequency of ngrams
count_values = ngrams.toarray().sum(axis=0)
# list of ngrams
vocab = c_vec.vocabulary_
df_ngram21 = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
df_ngram21['polarity'] = df_ngram21['bigram/trigram'].apply(lambda x: TextBlob(x).polarity)
df_ngram21['subjective'] = df_ngram21['bigram/trigram'].apply(lambda x: TextBlob(x).subjectivity)


df_ngram21



from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
tfidf_vectorizer = TfidfVectorizer(stop_words=stoplist, ngram_range=(2,3))
nmf = NMF(n_components=3)
pipe = make_pipeline(tfidf_vectorizer, nmf)
pipe.fit(df21["qns21"].dropna())
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += ", ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
print_top_words(nmf, tfidf_vectorizer.get_feature_names(), n_top_words=5)


peryear= df22["qns22"].dropna()
print(peryear)

for peryear 

"C:\Users\shanais.yuen\OneDrive - Seagroup\Desktop\AdsMonthlyReport_updated.zip"

"C:\Users\shanais.yuen\OneDrive - Seagroup\Desktop\Helpppppppp.xlsx"

df_ngram23.to_csv("C:/Users/shanais.yuen/Downloads/df_ngram23_march.csv", index = False)




"C:\Users\shanais.yuen\Downloads\Ads Package Analysis - 2.25 SSS Ads Packages.csv"















