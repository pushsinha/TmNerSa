# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:07:38 2018

@author: pushkar
"""

import string
import os
import codecs
import pandas as pd
import matplotlib.pyplot as plt



def plus3(row):
    if len(row['entity']) > 3 and row['entity'] is not None:
        return True
    else:
        return False
# 'df1' is entities in articles, 'df2' is entities in comments, 'result' is the overlapped entities in both.
# The all are pandas dataframes with columns ]entitiy, tag] in the beginning and [entity, tag, count, nametag] later.
df1 = pd.read_csv('ctags.csv')
df2 = pd.read_csv('atags.csv')
df1 = df1[df1.apply(plus3, axis = 1)]
df1 = df1.drop(columns = df1.columns[0])
df2 = df2.drop(columns = df2.columns[0])
df1['nametag'] = df1.apply(lambda row: row['entity'] + ',' + row['tag'], axis = 1)
df2['nametag'] = df2.apply(lambda row: row['entity'] + ',' + row['tag'], axis = 1)
df1 = df1.groupby(['entity','tag']).count().reset_index()
df2 = df2.groupby(['entity', 'tag']).count().reset_index()
#print(df1)
df1.columns = ['entity', 'tag', 'count']
df2.columns = ['entity', 'tag', 'count']
df1['nametag'] = df1.apply(lambda row: row['entity'] + ',' + row['tag'], axis = 1)
df2['nametag'] = df2.apply(lambda row: row['entity'] + ',' + row['tag'], axis = 1)
df1 = df1.sort_values(['count'], ascending = [False])
df2 = df2.sort_values(['count'], ascending = [False])
result = pd.merge(df1, df2, how = 'inner', on = ['entity','tag'])
result = result.drop(columns = [result.columns[3], result.columns[4], result.columns[5]])
result.columns = ['entity', 'tag', 'count']
result['nametag'] = result.apply(lambda row: row['entity'] + ',' + row['tag'], axis = 1)
result = result.sort_values(['count'], ascending = [False])
result.to_csv("ne_overlap.csv", index = False)

# selecting top 20 entities
df1 = df1.head(20)

# selecting 10-20 entities
labels = df1['nametag'].values[9:]
sizes = df1['count'].values[9:]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
print("Top entities in Comments\n\n")
df2 = df2.head(20)
labels = df2['nametag'].values[9:]
sizes = df2['count'].values[9:]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
print("Top entities in Articles\n\n")

result = result.head(20)
labels = result['nametag'].values[9:]
sizes = result['count'].values[9:]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')

plt.show()
print("Top entities of Articles in Comments\n\n")







