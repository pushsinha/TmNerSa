
# coding: utf-8
#%%matplotlib
import os
import nltk
import sys
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.ml.clustering import LDA
from pyspark.sql.types import *
from pyspark.sql import *
from openpyxl import Workbook
import re 
import struct
from struct import *
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StringType, ArrayType
import pandas as pd
from pyspark.ml.feature import StopWordsRemover, Tokenizer, CountVectorizer, HashingTF, IDF
from array import array
from pyspark.ml.feature import Word2Vec
from pyspark.sql.types import ArrayType, StringType
from openpyxl import Workbook
import matplotlib
matplotlib.use('Agg')

#import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xlsxwriter
from time import time
warnings.filterwarnings("ignore")
#%%matplotlib

import nltk
print ("the path is:", nltk.data.path)
nltk.data.path.append('file:///home/mbabaeva/nltk_data')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import os
import codecs
lemma = WordNetLemmatizer()

spark = SparkSession.builder          .appName("spark-nltk")          .getOrCreate()

data = spark.read.csv("gnm_comments.csv", header=True) #gives a dataframe
sc = spark.sparkContext

comment_text = data.select('comment_text')
comment_text=comment_text.replace(r'\\n\\n|\\n',' ')
comment_text=comment_text.replace(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""",' ')


def cleanup_text(text):
    nltk.data.path.append('/home/mbabaeva/nltk_data')   
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = text.lower().translate(translate_table)
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
    u'two', u'three', u'get', u'would', u'seem', u'want', u'hey', u'might', u'may', u'withot', u'with', u'also', 
    u'make', u'want', u'put', u'etc', u'actually', u'else', u'far', u'definitely', u'youll', u'didnt', u'isnt', 
    u'theres', u'since', u'able', u'maybe', u'sort', u'think', u'know', u'look', u'please', u'one', u'null', u'dont',
    u'could', u'unable', u'someday', u'best', u'better', u'someone', u'sure', u'lot', u'thank', u'also', u'anyone', u'really',
    u'something', u'give', u'years', u'use', u'all', u'ago', u'many', u'right', u'call', u'include', u'part', u'find',
    u'become', u'choose', u'chosen', u'as', u'back', u'good', u'right', u'see', u'good', u'even',u'first', u'low', u'less', u'much', u'another', u'mine',
    u'instead', u'will', u'never', u'ask', u'even', u'see', u'allow', u'still', u'that', u'you',
    u'disagree', u'obviously', u'self', u'bye', u'well', u'make', u'take', u'let',
    u'get', u'agree', u'leave', u'live', u'say', u'tell', u'understand', u'look', u'with', u'without',u'seem', u'bad',
    u'nothing', u'everything', u'give', u'long', u'think', u'show', u'last', u'run', u'day', u'great', 
    u'try', u'yes', u'no', u'live', u'right', u'perhaps', u'already', u'never', u'ever', u'just',   
    u'rather', u'however', u'real', u'least', u'good', u'bring', u'other', u'another', u'away', u'youre',     
    u'enough', u'want', u'mine', u'yours', u'hear', u'either', u'nor', u'neither', u'look', u'however',
    u'know', u'come', u'without', u'most', u'least', u'less', u'few', u'nah', u'bye', u'told'
    u'often', u'nah', u'bye', u'little', u'high', u'anything', u'wrong', u'though', u'always', u'every', 
    u'around', u'yet', u'little', u'like']
    
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    normalized = [lemma.lemmatize(word,'v') for word in words]
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in normalized]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    
    #normalized = " ".join(lemma.lemmatize(word,'v') for word in words)
    text_out = " ".join(lemma.lemmatize(word,'v') for word in text_out)
 
    
    return text_out


udf_cleantext = udf(cleanup_text, StringType())

clean_text = data.withColumn("clean_comm", udf_cleantext(data.comment_text))
#clean_text.select("clean_comm").show(3)

tokenizer = Tokenizer(inputCol="clean_comm", outputCol="tokens")

tokenized = tokenizer.transform(clean_text)
e = tokenized.select("clean_comm", "tokens")
#tokenized.select("clean_comm", "tokens").show(1)

remover = StopWordsRemover(inputCol="tokens", outputCol="tokens_Stop")

t = remover.transform(e)
tt = t.select('tokens_Stop')
#tt.show(2)

tt = tt.select('tokens_Stop')
tt = tt.limit(650000) #you can choose number or comments you want to run LDA on
tt.count()
#tt.toPandas().to_csv('cleaning.csv')  #Uncomment if oyu want to save a cleaned model


#Term Frequency Vectorization  - Option 1 (HashingTF)
#cv = HashingTF(inputCol="tokens_Stop", outputCol="Rawfeatures")
#featurizedData = cv.transform(tt)
#featurizedData.show(1)

# Term Frequency Vectorization  - Option 2 (CountVectorizer):

start = time()
cv = CountVectorizer(inputCol="tokens_Stop", outputCol="Rawfeatures")
cvmodel = cv.fit(tt)
featurizedData = cvmodel.transform(tt)
 
vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)
print ('used CV: {:.2f}s'.format(time()-start))

start = time()
idf = IDF(inputCol="Rawfeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData) # TFIDF
print ('used IDF: {:.2f}s'.format(time()-start))

#rescaledData.count()

p = rescaledData.select('features')
p = p.limit(650000) # you can choose number or comments you want to run LDA on
#p.count()
#p.show(3)

import threading
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')

#Calculating LDA:
start = time()
lda = LDA(k=20, maxIter=500)
model = lda.fit(p)
print ('used LDA: {:.2f}s'.format(time()-start))

#model.isDistributed()


#start = time()
#ll = model.logLikelihood(p)
#lp = model.logPerplexity(p)
#print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
#print("The upper bound on perplexity: " + str(lp))
#print ('used: {:.2f}s'.format(time()-start))

start = time()
# Describe topics.
topics = model.describeTopics(15)
print("The topics described by their top-weighted terms:")
topics.show(10)
print ('used to describe: {:.2f}s'.format(time()-start))

start = time()
# Shows the result
transformed = model.transform(p)
#transformed.show(30)
print ('used: {:.2f}s'.format(time()-start))

#Back To Words from vectors:
def indices_to_terms(vocabulary):
    def indices_to_terms(xs):
        return [vocabulary[int(x)] for x in xs]
    return udf(indices_to_terms, ArrayType(StringType()))

hh = topics.withColumn("topics_words", indices_to_terms(cvmodel.vocabulary)("termIndices"))
hh = hh.select("topic", "termWeights", "topics_words")

#hh.show(10)

yy = hh.toPandas().set_index('topic')

#Vizualization:
import matplotlib
matplotlib.use('Agg')

fiz=plt.figure(figsize=(15,30))
for i in range(15):
    plt.subplot(10,2,i+1)
    plt.title('topics '+str(i+1))
    sns.barplot(x=yy['termWeights'][i], y=yy['topics_words'][i], data=yy, label='Cities', palette='GnBu_d')
    plt.xlabel('probability')
    
#plt.show()
fiz.savefig('spark_charts_650000_20t.png')

yy = yy.to_excel('spark_topics_650000_20t.xlsx')

features_words = transformed.withColumn("comments_words", indices_to_terms(cvmodel.vocabulary)("features"))

#features_words.show(3)

#features_words.toPandas()

