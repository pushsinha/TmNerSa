#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 12:39:06 2018

@author: pushkarsinha
"""

import os
import nltk
import sys
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
#from pyspark.mllib.clustering import LDA
from pyspark.mllib.clustering import LDA, LDAModel
#from pyspark.ml.clustering import LDA
from pyspark.sql.types import *
from pyspark.sql import *
import re 
import struct
#import pyLDAvis
from struct import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StringType, ArrayType
import pandas as pd
from pyspark.ml.feature import StopWordsRemover, Tokenizer, CountVectorizer, HashingTF, IDF
from array import array
from pyspark.ml.feature import Word2Vec
from pyspark.sql.types import ArrayType, StringType

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import codecs

lemma = WordNetLemmatizer()

spark = SparkSession.builder \
         .appName("spark-nltk") \
         .getOrCreate()
#
data = spark.read.csv("../../../../SOCC/raw/gnm_comments.csv", header=True) #gives a dataframe
sc = spark.sparkContext
#
comment_text = data.select('comment_text')
comment_text=comment_text.replace(r'\\n\\n|\\n',' ')
comment_text=comment_text.replace(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""",' ')

def cleanup_text(text):
    
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = text.translate(translate_table)
    words = no_punct.split()
    #words = translate_table
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = ['per', u'one', u'tell', u'need', u'say', u'new', u'try', u'take', u'may', u'come', u'get', 
    u'two', u'three', u'get', u'would', u'seem', u'want', u'hey', u'might', u'may', u'without', u'with', u'also', 
    u'make', u'want', u'put', u'etc', u'actually', u'else', u'far', u'definitely', u'youll', u'didnt', u'isnt', 
    u'theres', u'since', u'able', u'maybe', u'sort', u'think', u'know', u'look', u'please', u'one', u'null', u'dont',
    u'could', u'unable', u'someday', u'someone', u'also', u'anyone', u'really',
    u'something', u'give', u'years', u'use', u'all', u'ago', u'right', u'call', u'include', u'part', u'find',
    u'become', u'choose', u'chosen', u'as', u'back', u'see', u'even',u'first', u'another', u'mine',
    u'instead', u'will', u'never', u'ask', u'even', u'see', u'allow', u'still', u'that', u'you',
    u'obviously', u'self', u'bye', u'well', u'make', u'take', u'let',
    u'get', u'leave', u'live', u'say', u'tell', u'understand', u'look', u'seem',
    u'nothing', u'everything', u'give', u'long', u'think', u'show', u'last', u'run', u'day', 
    u'try', u'yes', u'no', u'live', u'right', u'perhaps', u'already', u'never', u'ever', u'just',   
    u'rather', u'however', u'real', u'bring', u'other', u'another', u'away', u'youre',     
    u'enough', u'want', u'mine', u'yours', u'hear', u'either', u'nor', u'neither', u'look', u'however',
    u'know', u'come', u'without', u'least', u'nah', u'bye', u'told'
    u'often', u'anything', u'wrong', u'though', u'always', u'every', u'around', u'yet']
    
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word for word in stopwords]    
    
    normalized = [lemma.lemmatize(word,'v') for word in words]
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in normalized]                                       # Remove special characters
    text_out = [word for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    
    #normalized = " ".join(lemma.lemmatize(word,'v') for word in words)
    text_out = " ".join(lemma.lemmatize(word,'v') for word in text_out)
 

    #x = normalized.split()
    #y = [s for s in x if len(s) > 2]
    
    return text_out

def cleanup_arr(clean_comm):
    return clean_comm.split()
    
udf_cleantext = udf(cleanup_text, StringType())
udf_cleanarr = udf(cleanup_arr, ArrayType(StringType()))
clean_text = data.withColumn("clean_comm", udf_cleantext(data.comment_text))
clean_text1 = clean_text.select("clean_comm").limit(3)
clean_text =  clean_text.withColumn("clean_arr", udf_cleanarr(clean_text.clean_comm))
clean_text = clean_text.select("clean_arr")
#tokenizer = Tokenizer(inputCol="clean_comm", outputCol="tokens")
#tokenized = tokenizer.transform(clean_text)
#e = tokenized.select("clean_comm", "tokens")
#e1 = tokenized.select("tokens").limit(5)
remover = StopWordsRemover(inputCol="clean_arr", outputCol="tokens_Stop")
#
t = remover.transform(clean_text)
tt = t.select('tokens_Stop')
#tt.show(2)
#clean_text.rdd.saveAsTextFile("../../../SOCC/raw/cleaned_comments2.txt")
import numpy as np

np.savetxt("cleaned_comments.txt", tt.toPandas().values, fmt = '%s')

