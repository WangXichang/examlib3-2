# -*- utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math


# Interface standard score transform model
class StdScore(object):
    def __init__(self, modelname=''):
        self.modelname = modelname
        # self.__rawdf = None
        # self.__outdf = None
        # self.__scorefields = None

    def set_data(self, rawdf, scorefields=None):
        raise NotImplementedError()
        # define in subclass
        # self.__rawdf = rawdf
        # self.__scorefields = scorefields

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def report(self):
        raise NotImplementedError()

    def plot(self, *args):
        raise NotImplementedError()


# model for linear score transform on some intervals
class LstStdScore(StdScore):
    __doc__ = ''' LstModel:
    use linear standardscore transform from raw-score intervals
    to united score intervals
    '''

    def __init__(self):
        # intit __rawdf, __rawscorefields, __outdf
        # StdScore.__init__(self)
        self.__rawdf = None
        self.__outdf = None
        self.__scorefields = None
        # new properties for linear segment stdscore
        self.__stdScorePoints = []
        self.__rawScorePercentPoints = []
        self.__rawScorePoints__ = []
        self.__lstCoeff__ = {}
        self.__formulastr__ = ''
        # init super class
        super(LstStdScore, self).__init__('Segment Wise linear Transform Standard Score')
        return

    def set_data(self, rawdf, scorefields=None):
        if type(rawdf) == pd.Series:
            rawdf = pd.DataFrame(rawdf)
        elif type(rawdf) != pd.DataFrame:
            print('rawdf is not correct data set(DataFrame or Series)!')
            return
        if not scorefields:
            scorefields = [s for s in rawdf]
        if type(scorefields) != list:
            print('scorefields is not a list!')
            return
        if sum([1 if sf in rawdf else 0 for sf in scorefields]) != len(scorefields):
            print('scorefields is not correct(must in rawdf.columns)!')
            return
        self.__rawdf = rawdf
        self.__scorefields = scorefields

    def set_parameters(self, rawscorepercent=None, stdscorepoints=None):
        if (type(rawscorepercent) != list) | (type(stdscorepoints) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return
        if len(rawscorepercent) != len(stdscorepoints):
            print('len is not same for rawscorepoints and stdscorepoint list!')
            return
        self.__rawScorePercentPoints = rawscorepercent
        self.__stdScorePoints = stdscorepoints

    def run(self):
        for i in range(len(self.__scorefields)):
            self.__lstrun(self.__scorefields[i])

    def report(self):
        self.__formulastr__ = ['{0}*(x-{1})+{2}'.format(x[0], x[1], x[2])
                               for x in self.__lstCoeff__.values()]
        print('raw score percent points:\n\t', self.__rawScorePercentPoints)
        print('raw score points:\n\t', self.__rawScorePoints__)
        print('std score points:\n\t', self.__stdScorePoints)
        print('coefficent:\n', self.__lstCoeff__)
        print('formula: ', self.__formulastr__)

    def plot(self, mode='model'):
        if mode == 'model':
            self.plotmodel()
            return
        elif mode == 'raw':
            self.plotrawscore()
            return
        elif mode.lower() == 'std':
            self.plotlstscore()
            return
        print('no this mode!')

    # --------------property set end
    def __getcoeff(self):
        # the format is y = (y2-y1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1)/(x2 -x1)
        # bias  = y1
        # rawStartpoint = x1
        for i in range(1, len(self.__stdScorePoints)):
            coff = (self.__stdScorePoints[i] - self.__stdScorePoints[i - 1]) / \
                   (self.__rawScorePoints__[i] - self.__rawScorePoints__[i - 1])
            y1 = self.__stdScorePoints[i - 1]
            x1 = self.__rawScorePoints__[i - 1]
            coff = math.floor(coff*10000)/10000
            self.__lstCoeff__[i] = [coff, x1, y1]
        return

    def __getcore(self, x):
        for i in range(1, len(self.__stdScorePoints)):
            if x <= self.__rawScorePoints__[i]:
                return self.__lstCoeff__[i][0] * (x - self.__lstCoeff__[i][1]) + self.__lstCoeff__[i][2]
        return -1

    def __preprocess(self, field):
        if type(self.__rawdf) != pd.DataFrame:
            print('no dataset given!')
            return False
        if not self.__stdScorePoints:
            print('no standard score interval points given!')
            return False
        if not self.__rawScorePercentPoints:
            print('no score interval percent given!')
            return False
        if len(self.__rawScorePercentPoints) != len(self.__stdScorePoints):
            print('score interval for rawscore and stdscore is not same!')
            print(self.__stdScorePoints, self.__rawScorePercentPoints)
            return False
        if self.__stdScorePoints != sorted(self.__stdScorePoints):
            print('stdscore points is not in order!')
            return False
        if sum([0 if (x <= 1) & (x >= 0) else 1 for x in self.__rawScorePercentPoints]) == 1:
            print('raw score interval percent is not percent value !\n', self.__rawScorePercentPoints)
            return False
        # claculate _rawScorePoints
        if field in self.__rawdf.columns.values:
            __rawdfdesc = self.__rawdf[field].describe(self.__rawScorePercentPoints)
            # calculating _rawScorePoints
            self.__rawScorePoints__ = []
            for x in self.__rawScorePercentPoints:
                fname = '{0}%'.format(int(x * 100))
                self.__rawScorePoints__ += [__rawdfdesc.loc[fname]]
        else:
            print('error score field name!')
            print('not in '+self.__rawdf.columns.values)
            return False
        # calculate lstCoefficients
        self.__getcoeff()
        return True

    def __lstrun(self, scorefieldname):
        if not self.__preprocess(scorefieldname):
            print('fail to initializing !')
            return
        # create outdf
        self.__outdf = pd.DataFrame(self.__rawdf[scorefieldname])
        # transform score
        self.__outdf[scorefieldname + '_lst'] = \
            self.__outdf[scorefieldname].apply(self.__getcore)
        # self.__rawScoreField = scorefieldname
        return

    def plotrawscore(self):
        if not self.__scorefields:
            print('no field assign in rawdf!')
            return
        plt.figure('Raw Score figure')
        for sf in self.__scorefields:
            self.__rawdf.groupby(sf)[sf].count().plot(label=sf)
        return

    def plotlstscore(self):
        if not self.__scorefields:
            print('no field assign in rawdf!')
            return
        plt.figure('lst Score figure')
        for sf in self.__scorefields:
            self.__rawdf.groupby(sf + '_lst')[sf + '_lst'].count().plot(label=sf)
        return

    def plotmodel(self):
        plt.figure('Linear score Transform: {0}'.format(self.__scorefields))
        plen = len(self.__rawScorePoints__)
        plt.xlim(self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1])
        plt.ylim(self.__stdScorePoints[0], self.__stdScorePoints[plen-1])
        plt.plot(self.__rawScorePoints__, self.__stdScorePoints,
                 label='LSM x:rawscore y:stdscore')
        plt.plot([self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]],
                 [self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]])
        plt.show()
        return


def test_lstmodel():
    lst = LstStdScore()
    import py2ee_lib0901 as px
    rawdf = px.exp_scoredf_normal()
    stdpoints = [40, 55, 65, 80, 95, 110, 120]
    rawpercentpoints = [0, .15, .30, .50, .70, .85, 1.00]
    lst.set_data(rawdf, 'sf')
    lst.set_parameters(rawpercentpoints, stdpoints)
    lst.run()
    lst.report()
    # lst.plot('raw')   # plot raw score figure, else 'std', 'model'
    return lst


def test_lst_gk2015():
    lst = LstStdScore()
    rawdf = pd.read_csv(r'test\2015cj.csv',
                        dtype={'KM1': np.int16, 'KM2': np.int16, 'KM3': np.int16,
                               'KM4': np.int16, 'KLDM': np.str}
                        )
    # [0, .15, .30, .50, .70, .85, 1.00]    # std = 35
    rawpoints = [0, 0.03, 0.20, 0.50, 0.70, 0.73, 1]   # std = 20
    stdpoints = [50, 65, 80, 100, 120, 135, 150]  # 0-150 => 50-150
    lst.set_parameters(rawpoints, stdpoints)
    lst.set_data(rawdf, 'KM1')
    lst.run()
    lst.report()
    # lst.plotrawscore()
    # lst.plotlstscore()
    return lst
