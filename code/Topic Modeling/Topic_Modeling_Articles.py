
#from helper import *
#python -m pip install <name of packages/library>
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
from gensim.models.ldamodel import LdaModel as Lda
from gensim import corpora
import string
import os
import codecs
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim
import warnings
import matplotlib
from openpyxl import Workbook
matplotlib.use('Agg')
warnings.filterwarnings("ignore")
import gensim
import warnings
import pandas as pd
from pylab import figure
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Function to remove stop words from sentences, punctuation & lemmatize words. 
def clean(doc):
    exclude = set(string.punctuation)
    translate_table = dict((ord(char), None) for char in string.punctuation)
    no_punct = doc.lower().translate(translate_table)
    stop_free = " ".join([i for i in no_punct.split() if i not in stop])
    normalized = " ".join(lemma.lemmatize(word,'v') for word in stop_free.split())
    
    x = normalized.split()
    y = [s for s in x if len(s) > 2]
    return y

comments=pd.read_csv("gnm_articles.csv")


# Cleaning 
stop = set(stopwords.words('english'))
lemma = WordNetLemmatizer()

additional_stops = ['per','one','two','three']

stop.update(additional_stops)
cleanPost = [clean(doc) for doc in comments['article_text']]


# # add log for recording the model fitting data while training

from time import time
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                   filename='running.log',filemode='w')


#There are \n\n in data above, need to delete them
comments['article_text'] = comments['article_text'].replace(r'\\n\\n|\\n',' ', regex=True)

# Delete URL, website etc.
comments['article_text'] = comments['article_text'].replace(r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))""",' ', regex=True)

cleanPost = [clean(doc) for doc in comments['article_text']]

#cleanPost


print(len(cleanPost))
#print([len(x) for x in cleanPost])[1:1]
#print(cleanPost)

# Find the most frequent words and exclude NEUtral them. My bias!!! May be work more on that?
import itertools
flattened_cleanPost = list(itertools.chain(*cleanPost))

from collections import Counter
word_counts = Counter(flattened_cleanPost)
top_three = word_counts.most_common(200)
print(top_three)

most_fr = pd.DataFrame(top_three, columns=['words','count']).set_index('words')
most_fr.to_excel('most_frequent_Articles_50t.xlsx')

#CHART MOST FREQUENT
fiz=plt.figure(figsize=(8,8))
plt.title('Most frequent words in Articles')
sns.barplot(x='count', y=most_fr[1:20].index, data=most_fr[1:20], label='Cities', palette='Reds_d')
plt.xlabel('count')
fiz.savefig('50t_MF_Words_in_Articles.png')
#plt.show()

# Creating the term dictionary of our courpus, where every unique term is assigned an index. 
dictionary = corpora.Dictionary(cleanPost)
dictionary.save('dictionary.dict')
print (dictionary)

#After printing the most frequent words of the dictionary, I found that few words which are mostly content neutral words are also present in the dictionary. 
# These words may lead to modeling of “word distribution”(topic) which is neutral and do not capture any theme or content. 
# I made a list of such words and filtered all such words.
stoplist = set('agree didnt 690 881 ever just thats last first live let cannot amoung between talk little high once only or our ours out over own same she he the their too to from for have is are of once those this they we me been before after below both but by can cant did didn o does doesn having her his here hers herself i through because doing making during being been be about above after again against all am an and any are as at in sort their when what where which while who whom why will wont were very untill under up than must my myself into isnt its itself ll down anything like everything wrong 2016 2015 000 ooo either nor yes no long already obviously live right perhaps just rather however real least good bring other another away enough however yours hear either neither most few nah bye lol hi often though always every around yet give back allow still that said should would lot thank also unable able since could hey hi far might may need new old try take come get etc actually elase and better worse sure low even see less much disagree self bye obviously still that you more another mine instead never ask as so per will every instead never ask want leave bring give one tell say new try take may come get two three get would agree seem like want hey might may without also make want put etc actually else far definitely youll\' didnt\' isnt\' theres since able maybe without may suggestedsort never isredditmediadomain userreports far appreciate next think know need look please one null take dont dont\' want\' could able ask well best someone sure lot thank also anyone really something give years use make all ago people know many call include part find become '.split())
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_ids)

#build a corpus and save it for a future
# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above and save for a future use
doc_term_matrix = [dictionary.doc2bow(doc) for doc in cleanPost]

#corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)
print (len(doc_term_matrix))

start = time()
# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

#Creating the object for LDA model using gensim library & Training LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=20, iterations=500)
print ('used: {:.2f}s'.format(time()-start))

#Save a model
#      ldamodel.save('topic_articles.model')
#print(ldamodel.print_topics(num_topics=2, num_words=4))

ii = ldamodel.print_topics(num_topics=50, num_words=30)
df = pd.DataFrame(ii, columns=['id_topics', 'words']).set_index('id_topics')
df1 = df.to_csv('50_topics_on_articles.csv')
df2 = df.to_excel('50_topics_on_articles.xlsx')

#MAIN PLOT
viz = pyLDAvis.gensim.prepare(ldamodel, doc_term_matrix, dictionary)
pyLDAvis.save_html(viz, '50t_articles.html')

#Load a model
#ldamodel.load('topic_articles.model')

yo=ldamodel.get_document_topics(doc_term_matrix) #get topics on all 10339 articles
li = []
for i in range(len(comments)):
    
    new = pd.DataFrame(yo[i], columns=['id','prob']).sort_values('prob',ascending=False).drop(['prob'], axis=1)
    p = new.head(1).values.T.flatten().tolist()
    k = li.append(p)

df_topic_id = pd.DataFrame(li, columns=['topics_id'])
df_topic_id.index.name = 'article_text_id'
#df_topic_id.head(5) #topic_ids for all authors' articles 
#len(df_topic_id.index)

comment_data=pd.read_csv("gnm_articles.csv")

comment_data.shape #number of rows
len(comment_data.index)

comment_author= comment_data['author'].to_frame()
comment_author.head(3) #choose only authors
#len(comment_author.index)

p=comment_author.groupby(['author']).size().reset_index(name='counts') # count authors
p.head(3)
#len(p)

result = pd.concat([comment_author, df_topic_id], axis=1) #join two data frames: one is topic_ids and one is author name
result.head(3)
#len(result)

#showing all authors who write on this topic:
q = result.groupby(['topics_id']).sum()
q.head(3)

#how many authors per topic:
qq = result.groupby(['topics_id'])['author'].size().reset_index('topics_id')
#qq=qq.nlargest(20, 'author').reset_index()
qq1=qq.to_csv('50t_number_of_authors_per_topic.csv')
qq2 = qq.to_excel('50t_number_of_authors_per_topic.xlsx')
#qq.head(10)

len(qq['author'])

#FIGURE TOPIC CLUSTERING BY NUMBER ON AUTHORS
import numpy as np
import matplotlib.pyplot as plt

fir=plt.figure(figsize=(15,10))
plt.title('Topics clustering by number of Authors')
#np.random.seed(19680801)
N = 50
y = np.random.rand(N)
for i in range (0, len(qq)):
    #plt.scatter(qq['topics_id'][i], y[i], marker="o", s=qq['author'][i]*5, color=np.random.rand(3), alpha=0.8)
    plt.plot(qq['topics_id'][i], y[i], marker="o", markersize=qq['author'][i]*0.1, color=np.random.rand(3), alpha=0.8)

    plt.annotate(i, (qq['topics_id'][i], y[i]), size = 14)

fir.savefig('50t_articles_clustering.png')
#plt.show()

#Most Frequent Topics amoung articles
q = result.groupby(['topics_id']).size().reset_index(name='counts').sort_values('counts', ascending = False)
ty=q.to_csv('50t_most_frequent_topics_in_articles.csv')
tyy =q.to_excel('50t_most_frequent_topics_in_articles.xlsx')

#FIGURE TOPIC POPULARITY ON ARTICLES
fim=plt.figure(figsize=(20,10))
#sns.set(style="darkgrid")
#plt.subplot(5,2,1)
plt.title('Topic popularity on articles')
#sns.barplot(x=q.index, y= 'counts', data=q, label='Cities')
sns.countplot(x='topics_id', data=result, order = result['topics_id'].value_counts().index, palette="Greens_d") #set2
plt.xlabel('Topics ids')
plt.ylabel('Number of topics')

fim.savefig('50t_Topic_Popularity_on_Articles.png')
#plt.show()

#FIGURE SHOWING THE MOST POPULAR TOPICS:
import gensim
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

fip=plt.figure(figsize=(15,30))
for i in range(10):
    df=pd.DataFrame(ldamodel.show_topic((q.iloc[i]['topics_id'].astype(int)), topn=12), columns=['term','prob']).set_index('term')
    df = df.loc[df['prob'] >0.005]
    df
    plt.subplot(5,2,i+1)
    plt.title('topic '+str((q.iloc[i]['topics_id'].astype(int))))
    sns.barplot(x='prob', y=df.index, data=df, label='Cities', palette='Reds_d')
    plt.xlabel('probability')
fip.savefig('The_Most_Popular_Topics_on_Articles.png')
#plt.show()

p=result.groupby(['author','topics_id']).size().reset_index(name='counts') #count how many times the same author and topic_ids appears 
p.head(4)

#Most popular topic for each author:
t=p.groupby(['author'])['counts'].transform(max) == p['counts'] # take author and topic with a highest number of counts (appearence), meaning author and his topic_ids are more frequent
f=p[t]
print(f)
#len(f)


