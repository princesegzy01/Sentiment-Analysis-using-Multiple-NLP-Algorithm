#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 03:42:58 2018

@author: princesegzy01
"""

import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('NlP_Result_Transponse.csv',encoding = 'latin-1')

df = dataset[dataset['Kbest'] == 10]

grouped = dataset.groupby(['Classifier'])
for name,group in grouped:
    print(name)
    print(group)


