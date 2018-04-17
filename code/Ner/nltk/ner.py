# -*- coding: utf-8 -*-

from nltk.chunk import conlltags2tree, tree2conlltags
from nltk import word_tokenize, pos_tag, ne_chunk
import pandas as pd
import re

regex = re.compile('[^a-zA-Z]')

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
df = pd.DataFrame(columns = ['Evalue', 'part_of_speech', 'Egroup'])
nor = 0
with open('cleaned_articles.txt', 'r') as file:
    for sentence in file :
        sentence = regex.sub(' ', sentence)
        sentence = sentence.split()
        Csentence = ' '.join([ word for word in sentence if word != word.lower() ])   
        ne_tree = ne_chunk(pos_tag(word_tokenize(sentence)))
        iob_tagged = tree2conlltags(ne_tree)
        for entity in iob_tagged:
            df.loc[nor] = [ i for i in entity ]
            nor = nor + 1
        print(nor)
df.to_csv('ne.csv')