# -*- utf-8 -*-
#import pytable as pt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def chkFile(xmlTask):
    # get xml for check items
    pass
    # check filename
    pass
    # check file fields name and type and value range
    pass
    return

def readXml(xmlNmae):
    pass

def getDf(csvfile):
    return pd.read_csv(csvfile)
def getSeg(df,segColNames=[], segRange=[0,150]):
    if len(segColNames) == 0:
        _segColNames = df.columns.values
    else:
        _segColNames = segColNames
    dfseg = pd.DataFrame({'seg':[i for i in range(segRange[0],segRange[1]+1)]})
    for sn in _segColNames:
        dfgroup = df[df[sn]>=0].groupby(by=sn)[[sn]].count()
        segList = [int(dfgroup[dfgroup.index==i][sn].values[0]) \
                       if i in dfgroup.index else 0 for i in range(segRange[0],segRange[1]+1)]
        dfseg[sn+'_cou'] = segList
        dfseg[sn+'_sum'] = [sum(segList[0:i+1]) for i in range(len(segList))]
        sumMax = max(dfseg[sn+'_sum'])
        dfseg[sn+'_rat'] = [x/sumMax for x in dfseg[sn+'_sum']]
    dfseg.set_index('seg',drop=False,inplace=True)
    return dfseg

def describeDf(df,sfields = []):
    tdf = 0
    if len(sfields) == 0:
        sfields = [s for s in df.columns.values]
    for sf in sfields:
        if type(tdf) != pd.core.frame.DataFrame:
            tdf = pd.DataFrame(df[sf][df[sf]>0].describe()).T
        else:
            tdf = tdf.append(pd.DataFrame(df[sf][df[sf]>0].describe()).T)
    return tdf
# data preprocessing functions
# blank field to '-1' in a str-type field
def pp_blankField(df,fieldnames):
    for fieldname in fieldnames:
        df[fieldname]=df[fieldname].apply(lambda x: -1 if len(x.strip()) == 0 else int(eval(x)))
    return df

# data pp end

class stdScore(object):
    def __init__(self,df):
        self.df = df
        return
    def linearTrans(self,scoreField,rawScoreRange=[],stdScoreRange=[]):
        self.df[scoreField+'_lstd'] = self.df[scoreField]
        self.df[scoreField+'_lstd'] = [self._getLstdScore(x,rawScoreRange,stdScoreRange) \
                                        for x in self.df[scoreField]]
        self.df[scoreField+'_diff'] = self.df[scoreField+'_lstd'] - self.df[scoreField]
        return self.df
    def _getLstdScore(self, rs, rawScoreRange, stdScoreRange):
        for i in range(1,len(rawScoreRange)):
            if rs <= rawScoreRange[i]:
                ss = (stdScoreRange[i]-stdScoreRange[i-1]) / (rawScoreRange[i]-rawScoreRange[i-1]) * \
                     (rs - rawScoreRange[i-1]) + stdScoreRange[i-1]
                return ss
        return -1 #error range:rawscore is not in rawScoreRange

def testStdScore():
    rawScoreMean = 75
    rawScoreStd = 12
    rawScoreMax = 100
    stdScorePercentile = [.15,.30,.50,.70,.85]
    #stdScorePoints = [40,55,65,80,95,105,120] #model 40-120
    stdScorePoints = [50,65,80,100,120,135,150]  #model 50-150
    scoreDf = pd.DataFrame({'rs':[0]+[max(0,min(rawScoreMax,int(np.random.randn()*rawScoreStd+rawScoreMean))) \
                                      for x in range(rawScoreMax-1)]})
    p = scoreDf.describe(stdScorePercentile)
    plist = [0]+[p[p.index==v]['rs'].values[0] for v in ['15%','30%','50%','70%','85%']]+[100]
    ts = stdScore(scoreDf)
    #rdf = ts.linearTrans('rs',[0,55,62,70,78,85,100],[50,65,80,100,120,135,150])
    rdf = ts.linearTrans('rs',plist,stdScorePoints)
    rdfdes = rdf.describe(stdScorePercentile)
    x = [0]+rdfdes['rs'].ix[['15%','30%','50%','70%','85%']].values.tolist()+[100]
    y = stdScorePoints
    plt.figure('Linear Standard Score Model')
    plt.xlim([0,rawScoreMax])
    plt.ylim([stdScorePoints[0],max(stdScorePoints)])
    plt.plot(x,y,label="Linear StdScore for 50-150")
    plt.plot(x,x)
    return rdf,rdfdes


