Named Entity Recognition
Files ‘gnm_articles.csv’ and ‘gnm_comments.csv’ were the 2 files used.
‘cleaning_articles_ner.py’ and ‘cleaning_comments.py’ were used to clean both the articles and the comments data respectively.
These two use pyspark.
After cleaning, ‘ner.py’ can be used to tag the words in the cleaned data. On articles it takes ~2 days and on comments its 
has not been checked.
Alternatively there exists an NER project at ‘https://github.com/glample/tagger’ which can be used with the existing ‘english’
corpora to tag both articles’ and comments’ data. But before using it, make sure a C++ compiler exists in your machine without 
which the performance will be severely degraded.
Using the above project we get tagged data as ‘atags.txt’ (article tags) and ‘ctags.txt’ (comments tag). Further, using 
‘ner_clean_tab.py’ the generated tagged data is cleaned and kept in the form of csv files ‘atags.csv’(article tags) and 
‘ctags.csv’(comment tags).
For doing the first visualization (Figure 9, Figure 10, Figure 11) to get the top named entities ‘ner_final.py’ is used with 
‘atags.csv’(article tags) and ‘ctags.csv’(comment tags) as input files.The top entities in articles(Figure 9), 
comments(Figure 10), as well as the overlapping entities(Figure 11), in these two are displayed. The python file generates 
an output file (‘ne_overlap.csv’) which contains the overlapping entities in both articles and comments.
For the second visualization (Figure 12) use ‘common_author.py’ with ‘ne_overlap.csv’ as input. This visualization uses 
plotly as well as PCA, KMeans from SciKit learn. It displays the similar authors who mostly use the common entities in their 
texts. Any number / any type  of entity can be given for any desired visualizations.
