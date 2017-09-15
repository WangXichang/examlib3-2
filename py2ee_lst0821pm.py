# -*- utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# model for linear score transform on some intervals
class LstModel(object):
    ''' LstModel:
    use linear standardscore transform from raw-score intervals
    to united score intervals
    '''
    def __init__(self):
        self.__rawDf = None #pd.DataFrame({'rs':[x for x in range(150)]}) #example
        self.__rawDfdesc = None #self.__rawDf.describe()  #init value. not with percentpoints
        self.__rawScorePoints = []
        self.__stdScorePoints = []
        self.__rawScorePercentPoints = []
        self.__lstCoeff = {}
        self.__rawScoreField = ''
        return
    @property
    def lstRawScorePercentPoints(self):
        return self.__rawScorePercentPoints
    @lstRawScorePercentPoints.setter
    def lstRawScorePercentPoints(self,plist):
        self.__rawScorePercentPoints = plist
        return
    @property
    def lstStdScorePoints(self):
        return self.__stdScorePoints
    @property
    def lstStdScorePoints(self):
        return self.__stdScorePoints
    @lstStdScorePoints.setter
    def lstStdScorePoints(self,slist):
        if sorted(slist) != slist:
            print('standard score interval is not ascending !')
        self.__stdScorePoints = slist
        return
    @property
    def lstRawScoreDataFrame(self):
        return self.__rawDf
    @lstRawScoreDataFrame.setter
    def lstRawScoreDataFrame(self,df):
        theType = type(df)
        if not (theType in [pd.DataFrame, pd.Series]):
            print('input dataset is not dataframe or series!')
            return
        elif theType == pd.Series:
            self.__rawDf = pd.DataFrame(df)
        else:
            self.__rawDf = df
        #self.__rawDfdesc = self.__rawDf.describe() #not suitable here
    @property
    def lstFormula(self):
        return self.__lstCoeff
    @property
    def lstRawScorePoints(self):
        return self.__rawScorePoints
    #--------------property end
    def __calcCoeff(self):
        # the format is y = (y2-y1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1)/(x2 -x1)
        # bias  = y1
        # rawStartpoint = x1
        for i in range(1, len(self.__stdScorePoints)):
            coff = (self.__stdScorePoints[i] - self.__stdScorePoints[i - 1]) / \
                   (self.__rawScorePoints[i] - self.__rawScorePoints[i-1])
            y1 = self.__stdScorePoints[i - 1]
            x1 = self.__rawScorePoints[i-1]
            self.__lstCoeff[i] = [coff, x1, y1]
        return
    def __calcScore(self,x):
        for i in range(1, len(self.__stdScorePoints)):
            if x <= self.__rawScorePoints[i]:
                return self.__lstCoeff[i][0] * (x - self.__lstCoeff[i][1]) + self.__lstCoeff[i][2]
        return -1
    def __preprocess(self,scoreFieldName):
        if type(self.__rawDf) != pd.DataFrame:
            print('no dataset given!')
            return False
        if self.__stdScorePoints == []:
            print('no standard score interval points given!')
            return False
        if self.__rawScorePercentPoints == []:
            print('no score interval percent given!')
            return False
        if len(self.__rawScorePercentPoints) != len(self.__stdScorePoints):
            print('score interval for rawscore and stdscore is not same!')
            print(self.__stdScorePoints, self.__rawScorePercentPoints)
            return False
        if self.__stdScorePoints != sorted(self.__stdScorePoints):
            print('stdscore points is not in order!')
            return False
        if sum([0 if (x<=1) & (x>=0) else 1 for x in self.__rawScorePercentPoints]) == 1:
            print('raw score interval percent is not percent value !\n', self.__rawScorePercentPoints)
            return False
        # claculate _rawScorePoints
        if scoreFieldName in self.__rawDf.columns.values:
            self.__rawDfdesc = self.__rawDf[scoreFieldName].describe(self.__rawScorePercentPoints)
            # calculating _rawScorePoints
            self.__rawScorePoints = []
            for x in self.__rawScorePercentPoints:
                fname = '{0}%'.format(int(x * 100))
                self.__rawScorePoints += [self.__rawDfdesc.loc[fname]]
        else:
            print('error score field name!')
            print('not in '+self.__rawDf.columns.values)
            return False
        # calculate lstCoefficients
        self.__calcCoeff()
        return True
    def transform(self, scoreFieldName):
        if not self.__preprocess(scoreFieldName):
            print('fail to initializing !')
            return
        #print(self.__rawScorePoints)
        #print(self._lstCoeff)
        # transform score
        self.__rawDf[scoreFieldName+'_lst'] = \
            self.__rawDf[scoreFieldName].apply(self.__calcScore)
        self.__trans = True
        self.__rawScoreField = scoreFieldName
        return
    def plotrawscore(self):
        if self.__rawScoreField != '':
            plt.figure('Raw Score figure')
            self.__rawDf.groupby(self.__rawScoreField)[self.__rawScoreField].count().\
                plot(label=self.__rawScoreField.__str__())
        else:
            print('no raw score field assign!')
        return
    def plotlstscore(self):
        if self.__rawScoreField != '':
            plt.figure('lst Score figure')
            self.__rawDf.groupby(self.__rawScoreField + '_lst')[self.__rawScoreField + '_lst'].count().\
                plot(label=self.__rawScoreField.__str__())
        else:
            print('no raw score field assign!')
        return
    def plotlstmodel(self):
        plt.figure('Linear score Transform')
        plt.plot(self.__rawScorePoints, self.__stdScorePoints, \
                 label = 'LSM x:rawscore y:stdscore')
        plt.show()
        return
    def report(self):
        print('raw score percent points:\n', self.__rawScorePercentPoints)
        print('raw score points:\n',self.__rawScorePoints)
        print('std score points:\n', self.__stdScorePoints)
        print('coefficent:\n', self.__lstCoeff)
def expLst():
    lst = LstModel()
    lst.lstRawScoreDataFrame = expDf()
    lst.lstStdScorePoints = [40, 55, 65, 80, 95, 110, 120]
    lst.lstRawScorePercentPoints = [0, .15, .30, .50, .70, .85, 1.00]
    lst.transform('sf')
    #lst.plotlstmodel()
    #lst.plotrawscore()
    #lst.plotlstscore()
    lst.report()
    return lst.lstRawScoreDataFrame

def expDf():
    mean=70; std = 10; maxscore = 150; minscore = 0; samples = 1000
    df = pd.DataFrame({'sf':[max(minscore, min(int(np.random.randn(1)*std + mean),maxscore)) \
                                                for x in range(samples)]})
    return df