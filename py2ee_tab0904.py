# -*- coding:utf-8 -*- 

from texttable import Texttable
import numpy as np
import pandas as pd


def maketab(df, *args, **kwargs):
    print(__tabText(df, *args, **kwargs))
    return

# -----------------------new texttable for this package
# have modified texttable for produce normal line with chinese char
# texttable function --
# def len(iterable):
#   ...
#   before modifying:
#   w = unicodedata.east_asian_width
#   return sum([w(c) in 'WF' and 2 or 0 if unicodedata.combining(c) else 1 for c in unicode_data])
#   after modifying:
#   return sum([2 if uchar >= u'\u4e00' and uchar <= u'\u9fa5' else 1 for uchar in unicode_data])
# -------------------------improve texttable notes end
# columnsFormat = [[columnsNo,Width,dtype,,align],...]
#   style:
#           't',  # text
#           'f',  # float (decimal)
#           'e',  # float (exponent)
#           'i',  # integer
#           'a'])  # automatic
#   Api:    listTable
# --------------------------notes for parameters format

def __tabText(df, colWidth=None, colNames=None, vLine=True, hLine=None,
         columnsFormat = None):
    if type(df) is not pd.DataFrame:
        print('Warning:\n',type(df),'\n',df)
        print('input data is not DataFrame!')
        return
    colNum = df.columns.__len__()
    rowNum = df.__len__()
    table = Texttable()
    if vLine == False:
        table.set_deco(Texttable.HEADER)
    table.set_cols_align(["l"] * colNum)
    table.set_cols_valign(["m"] * colNum)
    table.set_chars(["-","|","+","="])
    table.set_cols_dtype(['t'] * colNum)
    if not colWidth:
        colWidth = {}
    elif type(colWidth) != dict:
        print('colWidth is not dict type!')
        return
    defaultWidth = [10] * colNum
    if len(colWidth) > 0:
        for k in colWidth:
            if (type(k) == int) & (k in range(1,colNum+1)):
                defaultWidth[k-1] = colWidth[k]
            else:
                print('colwidth is error!')
    table.set_cols_width(defaultWidth)
    if colNames != None:
        headNames=colNames
    else:
        headNames= [s for s in df.columns]
    rowAll= [headNames] \
            + [list(df.values[i]) for i in range(len(df))]
    table.add_rows(rowAll)
    rr = table.draw()
    return rr

# chinese char processing function
def reduceBlankforCstr(cStr):
    lStr = cStr.split(sep='\n')
    rStr = ''
    for s in lStr:
        if s == '':
            continue
        lls = s.split(sep='|')
        nlls = [ss if _count_chineseChar(ss)==0 \
                else ss[0:len(ss)-_count_chineseChar(ss)] \
                for ss in lls]
        rStr = rStr + ''.join(['|'+r if r != '' else '' for r in nlls]+['|'])+'\n'
    return rStr
def _is_chineseChar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False
def _count_chineseChar(uStr):
    return sum([1 if _is_chineseChar(c) else 0 for c in uStr])
def strWidth(uStr):
    return len(uStr)+_count_chineseChar(uStr)

# create data example
def expDfwithcc():
    r= pd.DataFrame({"family":["王莽族","张春秋大家族人很多","郭子仪"], \
                     "level":[1,2,3], \
                     "name":["Robit欧洲人","Smith美国人","希腊Garisson"]})
    return r
def expSr():
    r=pd.Series(['Robit','Smith','Family'],index=range(3),name='name')
    return r
def expNormalDf(colNum=1 , size = 10, delta= 1, mu= 0):
    columnsList = ['v%d'%i for i in range(colNum)]
    valList = {columnsList[i] : np.random.randn(size)*delta + mu for i in range(colNum)}
    return pd.DataFrame(valList)

def testTable(df):
    #rowAll= [list(df.columns.values)] \
    #    +[list(df.ix[ri]) for ri in range(len(df))]
    table = Texttable()
    table.set_cols_align(["l", "r", "c"])
    table.set_cols_valign(["t", "m", "b"])
    old=1
    if old==1:
        rowList = [["Name", "Age", "Nickname"],\
                  ["Mr\n胡大山\n胡树林", 32, "Xav'"],\
                  ["Mr\\n陈清器\\n刘为级", 1, "Baby"],\
                  ["Mme\\nLouise\\n德里市", 28, "Lou\\n\\nLoue"]]
        table.add_rows(rowList)
    else:
        table.add_rows("")
    print(table.draw() + "\n")
    return
