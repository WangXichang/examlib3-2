# -*- utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from texttable import Texttable

def exp_scoredf_normal(mean=70, std = 10, maxscore = 100, minscore = 0, samples = 100000):
    df = pd.DataFrame({'sf':[max(minscore, min(int(np.random.randn(1)*std + mean),maxscore))
                             for x in range(samples)]})
    return df

def create_NormalDistTable(size= 400, std= 1, mean= 0, stdNum=4):
    '''
    create normal distributed data(pdf,cdf) with preset std,mean,samples size
    in [-stdNum * std, std * stdNum]
    :param size: samples number for create normal distributed PDF and CDF
    :param std:  standard difference
    :param mean: mean value
    :param stdNum: used to define data range [-std*stdNum, std*stdNum]
    :return: DataFrame['sv':stochastic var value, 'pdf':pdf data, 'cdf':cdf data]
    '''
    interval = [mean - std*stdNum, mean + std*stdNum]
    step = (2 * std * stdNum) / size
    x = [mean + interval[0] + v*step for v in range(size+1)]
    nplist = [1/(math.sqrt(2*math.pi)*std) * math.exp(-(v - mean)**2 / (2 * std**2)) for v in x]
    ndf = pd.DataFrame({'sv':x, 'pdf':nplist})
    ndf['cdf'] = ndf['pdf'].cumsum() * step
    return ndf

def read_normaltable(readrows= None):
    '''
    read and create normal distributed N(0,1) data from a high pricise data 100w samples in [-6, 6]
    used to make low precise data that is suitable to some applications
    :param readrows=> rows to read from 100w samples with -6 and 6
    :return=> normal data dataframe['No','sv','pdf','cdf']
    '''
    if type(readrows) == int:
        skipv = int(1000000/readrows)
        skiprowlist = [x if x % skipv != 0 else -1 for x in range(1000000)]
        skiprowlist = list(set(skiprowlist))
        if -1 in skiprowlist:
            skiprowlist.remove(-1)
        if 1 in skiprowlist:
            skiprowlist.remove(1)
    else:
        skiprowlist = []
    nt = pd.read_csv('normaldist100w.csv',
                     dtype={'sv': np.float32, 'cdf': np.float64, 'pdf': np.float64},
                     skiprows= skiprowlist)
    return nt

# --- print text table for DataFrame use texttable.py
def disp(df, colWidth=None, colNames=None, firstrows=-1):
    if type(df) is not pd.DataFrame:
        print(df)
        print('df is not DataFrame!')
        return
    colNum = df.columns.__len__()
    rowNum = df.__len__()
    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l"]*colNum)#require three  columns
    table.set_cols_valign(["m"]*colNum)
    table.set_chars(["-","|","|","-"])
    table.set_cols_dtype(['a']*colNum)
    if colWidth==None:
        colWidth= [10]*colNum
        table.set_cols_width(colWidth)
    else:
        table.set_cols_width(colWidth)
    if colNames!=None:
        headNames=colNames
    else:
        headNames= [s for s in df.columns]
    #rowAll= [headNames]
    if (type(firstrows) == int) & firstrows>0:
        rowNum = min(firstrows, rowNum)
    rowAll = [list(df.ix[ri]) for ri in range(rowNum)]
    #print(colWidth)
    #print(rowAll)
    table.add_rows(rowAll)
    print(table.draw() + "\n")
    return

