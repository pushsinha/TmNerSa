#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 03:47:55 2018

@author: pushkarsinha
"""

import pandas as pd
import re


df1 = pd.DataFrame(columns = ['entity', 'tag'])
df2 = pd.DataFrame(columns = ['entity', 'tag'])
nor = 0
regex1 = re.compile('(.+?)__O')
regex2 = re.compile('[^a-zA-Z]')
with open("../../ctags.txt", 'r') as file:
    for line in file:
        temp = regex1.sub('', line).split()
        for value in temp :
            row = value.split("__")
            row[0] = regex2.sub('', row[0])
            df.loc[nor] = row
            nor = nor + 1

nor = 0
with open("../../atags.txt", 'r') as file:
    for line in file:
        temp = regex1.sub('', line).split()
        for value in temp :
            row = value.split("__")
            row[0] = regex2.sub('', row[0])
            df.loc[nor] = row
            nor = nor + 1
df.to_csv('atags.csv')