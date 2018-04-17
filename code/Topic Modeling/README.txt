
TOPIC MODELING:

Please go to the website and download files from Linguistic Department with all data to work on :
https://researchdata.sfu.ca/islandora/object/islandora%3A9109
You need to download SFU-_SOCC.zip with all available files

First file, Topic_Modeling_Comments_Spark.py: a program uses a gnm_comments.csv file with 663,173 comments.
This program is written in PySpark.
You can run it with the following query: spark-submit Topic_Modeling_Comments_Spark.py 2>/dev/null
As a result you will get a spark_topics_650000_20t.xlsx file with 20 topics and 15 main words(terms) in those topics 
executed on 650000 comments.
Also you will be able to see how much time it takes to execute LDA algorithm.
Plus you will get a visualization picture such as 15 most important topics with most important words(terms) based on 
appearance frequency in those topics. 
spark_topics_650000_20t.xlsx
spark_charts_650000_20t.png

Second file, Topic_Modeling_Comments.py. Because of a lack of visualization options with PySpark it was decided to write 
the same program but just in Python to do exactly the same but with more visualization options. Though to run this 
programm we first cleaned out comments using SPARK to accelerate execution. So First please run the following programm 
Topic_Modeling_Comments_Cleaning_Spark.py on gnm_comments.csv file. This program prepares data to run LDA algorithm on it
performing the following normalization, lemmatization, cleaning from stop words, and tokenization.
As the result a program generate cleaning.csv file. With this cleaning the size of the file reduces twice from 
the original one. Now you can use it to run Topic_Modeling_Comments.py with the next command: python3 Topic_Modeling_Comments.py
As the result of running this program you will get:
- A file and picture with 30 most frequent words in comments:
  most_frequent_50t.xlsx
  50t_MF.png
- Then after executing additional cleaning on unnecessarily words corpus is created and LDA model is running resulting in
creating a file: 50t.csv (50t.xlsx) which is having all 50 topics for our comments.
- Visualization of all those 50 topics using an interactive PyLDAvis.gensim model, 50t.html. Where you can navigate on every topic and 
see most frequent words which appeared in this specific topic.
- The last you will get a 50t.png picture of all topics with top 15 most important words.

Third file, Topic_Modeling_Articles.py. We were using just Python with any PySpark because it consists of only 10,339 articles which were not required using Spark.
A program is using gnm_article.csv file which can be found in the same SFU-_SOCC.zip.
First a program is cleaning an article_text column and preparing it by performing normalization, lemmatization, 
cleaning from stop words, and tokenization to be able to run LDA on it.
As a result most_frequent_Articles_50t.xlsx and 50t_MF_Words_in_Articles.png are generated showing the most frequent words 
in the all articles.
After that more cleaning is performed eliminating words which are not relevant.
Next step is to create a corpus and run a gensim LDA model. As a result 50_topics_on_articles.csv (50_topics_on_articles.xlsx) file is created with 50 topics on articles.
As a visualisation tool again interactive pyLDAvis is used. 50t_articles.html.
With get_document_topics(doc) function we were able to extract and create a list of all topics related to their articles. One by one.
Then we merge this list of topics with a column of author who wrote that specific article.
Now we were able to cluster our 50 topics based on number of authors whose articles were related to those topics.
This 50t_number_of_authors_per_topic.csv file showing the number of authors per each topic.
After that it was not difficult to visualize that result 50t_articles_clustering.png. The bigger a circle the more authors 
who wrote their articles on that topic. Also another picture was created The_Most_Popular_Topics_on_Articles.png which shows 
a number of authors who wrote on that article, most popular topics in the given data of articles.
