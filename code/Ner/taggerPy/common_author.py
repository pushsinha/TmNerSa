#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 05:19:31 2018

@author: pushkarsinha
"""

import string
import os
import codecs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly as py
import plotly.graph_objs as go

result = pd.read_csv("ne_overlap.csv")
dfCList = ['author']
dfCList = dfCList + list(set(result['entity'].values[9:]))
dfC = pd.DataFrame(columns = dfCList)
def prepDFC(row):
    temp = []
    temp.append(row['author'])
    for e in dfCList[1:]:
        temp.append(row['article_text'].count(e))
    dfC.loc[prepDFC.nor] = temp
    prepDFC.nor = prepDFC.nor + 1
    
prepDFC.nor = 0
    
dfaRaw =  pd.read_csv('../../../../SOCC/raw/gnm_articles.csv')
dfaRaw = dfaRaw.drop(columns = ['article_id', 'title', 'article_url','published_date','ncomments', 'ntop_level_comments'])
dfaRaw.apply(prepDFC, axis = 1)
dfC1 = dfC.iloc[:, 1:]
X = dfC1.values
pca = PCA(n_components = 2).fit(X)
pca_2d = pca.transform(X)
kmeans = KMeans(init = 'k-means++').fit(pca_2d)
c = pd.Series(kmeans.labels_)
temp = []
for f in pca_2d[:, :1]:
    temp.append(f[0])
x = pd.Series(temp)
temp = []
for s in pca_2d[:, 1:]:
    temp.append(s[0])
y = pd.Series(temp)
dfaRaw['class1'] = c.values
dfaRaw['x'] = x.values
dfaRaw['y'] = y.values
dfaRaw.to_csv("dfaRaw.csv", index = False)
df = pd.read_csv("dfaRaw.csv")
print(dfaRaw)

c0 = df[df.class1 == 0]
c1 = df[df.class1 == 1]
c2 = df[df.class1 == 2]
c3 = df[df.class1 == 3]
c4 = df[df.class1 == 4]
c5 = df[df.class1 == 5]
c6 = df[df.class1 == 6]
c7 = df[df.class1 == 7]



data = [
    go.Scatter(
        x=c0['x'].tolist(),
        y=c0['y'].tolist(),
        mode='markers',
        text=c0['author'].tolist()
    ),
    go.Scatter(
        x=c1['x'].tolist(),
        y=c1['y'].tolist(),
        mode='markers',
        text=c1['author'].tolist()
    ),
    go.Scatter(
        x=c2['x'].tolist(),
        y=c2['y'].tolist(),
        mode='markers',
        text=c2['author'].tolist()
    ),
    go.Scatter(
        x=c3['x'].tolist(),
        y=c3['y'].tolist(),
        mode='markers',
        text=c3['author'].tolist()
    ),
    go.Scatter(
        x=c4['x'].tolist(),
        y=c4['y'].tolist(),
        mode='markers',
        text=c4['author'].tolist()
    ),
    go.Scatter(
        x=c5['x'].tolist(),
        y=c5['y'].tolist(),
        mode='markers',
        text=c5['author'].tolist()
    ),
    go.Scatter(
        x=c6['x'].tolist(),
        y=c6['y'].tolist(),
        mode='markers',
        text=c6['author'].tolist()
    ),
    go.Scatter(
        x=c7['x'].tolist(),
        y=c7['y'].tolist(),
        mode='markers',
        text=c7['author'].tolist()
    )
]
layout = go.Layout(
        title = 'Clustered(similar) Authors for the top entities (Trudeau, Justin, Harper, Margaret, John, Ontario etc.) they use most in their text !! ',
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.offline.plot(fig, validate=False, filename='WronglyPredicted.html')
#plot_url = py.plot(fig, filename='text-chart-basic')