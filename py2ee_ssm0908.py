# -*- utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
# import py2ee_seg0906 as pg
import py2ee_lib as pl


# test Score model
def exp_scoredf_normal(mean=70, std=10, maxscore=100, minscore=0, samples=100000):
    return pd.DataFrame({'sf': [max(minscore, min(int(np.random.randn(1) * std + mean), maxscore), -x)
                         for x in range(samples)]})


def test_model(name='plt', df=None, fieldname='sf'):
    if type(df) != pd.DataFrame:
        scoredf = exp_scoredf_normal()
    else:
        scoredf = df
    if name == 'plt':
        pltmodel = PltScoreModel()
        rawpoints = [0, 0.023, 0.169, 0.50, 0.841, 0.977, 1]   # normal ratio
        # rawpoints = [0, .15, .30, .50, .70, .85, 1.00]    # ajust ratio
        stdpoints = [40, 50, 65, 80, 95, 110, 120]  # std=15
        pltmodel.set_data(scoredf, [fieldname])
        pltmodel.set_parameters(rawpoints, stdpoints)
        pltmodel.run()
        pltmodel.report()
        pltmodel.plot('raw')   # plot raw score figure, else 'std', 'model'
        return pltmodel
    if name == 'z':
        zm = Zscore()
        zm.set_data(scoredf, [fieldname])
        zm.set_parameters(stdnum=4, maxscore=150, minscore=0)
        zm.run()
        zm.report()
        return zm
    if name == 't':
        tm = Tscore()
        tm.set_data(scoredf, [fieldname])
        tm.set_parameters(tscore_max=150, tscore_min=0)
        tm.run()
        tm.report()
        return tm


# Interface standard score transform model
class ScoreTransformModel(object):
    __doc__ = '''
    转换分数是原始分数通过特定模型到预定义标准分数量表的映射结果。
    标准分常模是将原始分数与平均数的距离以标准差为单位表示出来的量表。
    因为它的基本单位是标准差，所以叫标准分数,也可以称为转换分数。
    常见的标准分常模有：z分数、Z分数、T分数、标准九分数、离差智商（IQ）等。
    标准分常模分数均是等距分数，虽然不同类型的常模其平均数和标准差不同，但均可用离均值来表示。
    标准分数可以通过线性转换，也可以通过非线性转换得到，
    由此可将标准分数分为线性转换的标准分数与非线性转换的标准分数。'''

    def __init__(self, modelname='Test Score Transform Model'):
        self.modelname = modelname
        self.rawdf = None
        self.outdf = None
        self.scorefields = None

    def set_data(self, rawdf=None, scorefields=None):
        raise NotImplementedError()
        # define in subclass
        # self.__rawdf = rawdf
        # self.__scorefields = scorefields

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()

    def check_data(self):
        if type(self.rawdf) != pd.DataFrame:
            print('rawdf is not dataframe!')
            return False
        if (type(self.scorefields) != list) | (len(self.scorefields) == 0):
            print('no score fields assigned!')
            return False
        return True

    def check_parameter(self):
        return True

    def run(self):
        if not self.check_data():
            print('check data find error!')
            return False
        if not self.check_parameter():
            print('check parameter find error!')
            return False
        return True

    def report(self):
        raise NotImplementedError()

    def plot(self, mode='raw'):
        # implemented plot_out, plot_raw score figure
        if mode == 'out':
            self.__plotoutscore()
        elif mode == 'raw':
            self.__plotrawscore()
        else:
            return False
        return True

    def __plotoutscore(self):
        if not self.scorefields:
            print('no field:{0} assign in {1}!'.format(self.scorefields, self.rawdf))
            return
        plt.figure(self.modelname+' score figure')
        for sf in self.scorefields:
            for osf in self.outdf.columns.values:
                if sf+'_' in osf:  # find sf_outscore field
                    self.outdf.groupby(osf)[osf].\
                                       count().plot(label=self.scorefields.__str__())
        return

    def __plotrawscore(self):
        if not self.scorefields:
            print('no field assign in rawdf!')
            return
        plt.figure('Raw Score figure')
        for sf in self.scorefields:
            self.rawdf.groupby(sf)[sf].count().plot(label=f'{self.scorefields}')
        return


# model for linear score transform on some intervals
class PltScoreModel(ScoreTransformModel):
    __doc__ = ''' PltModel:
    use linear standardscore transform from raw-score intervals
    to united score intervals
    '''

    def __init__(self):
        # intit rawdf, scorefields, outdf, modelname
        super(PltScoreModel, self).__init__('Pieceise Linear Transform Model')
        # new properties for linear segment stdscore
        self.__stdScorePoints = []
        self.__rawScorePercentPoints = []
        self.__rawScorePoints__ = []
        self.__pltCoeff = {}
        self.__formulastr = ''
        self.modelname = 'plt'
        return

    def set_data(self, dfname=None, scorefieldnamelist=None):
        # check and set rawdf
        if type(dfname) == pd.Series:
            self.rawdf = pd.DataFrame(dfname)
        elif type(dfname) == pd.DataFrame:
            self.rawdf = dfname
        else:
            print('rawdf set fail!\n not correct data set(DataFrame or Series)!')
        # check and set scorefields
        if not scorefieldnamelist:
            self.scorefields = [s for s in dfname]
        elif type(scorefieldnamelist) != list:
            print('scorefields set fail!\n not a list!')
            return
        elif sum([1 if sf in dfname else 0 for sf in scorefieldnamelist]) != len(scorefieldnamelist):
            print('scorefields set fail!\n field must in rawdf.columns!')
            return
        else:
            self.scorefields = scorefieldnamelist

    def set_parameters(self, rawscorepercent=None, stdscorepoints=None):
        if (type(rawscorepercent) != list) | (type(stdscorepoints) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return
        if len(rawscorepercent) != len(stdscorepoints):
            print('len is not same for rawscorepoints and stdscorepoint list!')
            return
        self.__rawScorePercentPoints = rawscorepercent
        self.__stdScorePoints = stdscorepoints

    def check_parameter(self):
        if not self.scorefields:
            print('no score field assign in scorefields!')
            return False
        if (type(self.__rawScorePercentPoints) != list) | (type(self.__stdScorePoints) != list):
            print('rawscorepoints or stdscorepoints is not list type!')
            return False
        if (len(self.__rawScorePercentPoints) != len(self.__stdScorePoints)) | \
                len(self.__rawScorePercentPoints) == 0:
            print('len is 0 or not same for raw score percent and std score points list!')
            return False
        return True

    def run(self):
        if not super().run():
            return
        # create outdf
        self.outdf = self.rawdf[self.scorefields]
        # transform score on each field
        for i in range(len(self.scorefields)):
            self.__pltrun(self.scorefields[i])

    def report(self):
        self.__formulastr = ['{0}*(x-{1})+{2}'.format(x[0], x[1], x[2])
                             for x in self.__pltCoeff.values()]
        print('raw score percent points:', self.__rawScorePercentPoints)
        print('raw score points:', self.__rawScorePoints__)
        print('std score points:', self.__stdScorePoints)
        print('coefficent:', self.__pltCoeff)
        print('formula:', self.__formulastr)

    def plot(self, mode='model'):
        if mode == 'model':
            self.__plotmodel()
        elif not super().plot(mode):
            print('no this mode "%s"' % mode)

    # --------------property set end
    def __getcoeff(self):
        # the format is y = (y2-y1)/(x2 -x1) * (x - x1) + y1
        # coeff = (y2-y1)/(x2 -x1)
        # bias  = y1
        # rawStartpoint = x1
        for i in range(1, len(self.__stdScorePoints)):
            if (self.__rawScorePoints__[i] + self.__rawScorePoints__[i-1]) < 0.1**6:
                print('raw score percent is not differrentiable,{}-{}'.format(i, i-1))
                return False
            coff = (self.__stdScorePoints[i] - self.__stdScorePoints[i - 1]) / \
                   (self.__rawScorePoints__[i] - self.__rawScorePoints__[i - 1])
            y1 = self.__stdScorePoints[i - 1]
            x1 = self.__rawScorePoints__[i - 1]
            coff = math.floor(coff*10000)/10000
            self.__pltCoeff[i] = [coff, x1, y1]
        return True

    def __getcore(self, x):
        for i in range(1, len(self.__stdScorePoints)):
            if x <= self.__rawScorePoints__[i]:
                return self.__pltCoeff[i][0] * (x - self.__pltCoeff[i][1]) + self.__pltCoeff[i][2]
        return -1

    def __preprocess(self, field):
        # check format
        if type(self.rawdf) != pd.DataFrame:
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
        if sum([0 if (x <= 1) & (x >= 0) else 1 for x in self.__rawScorePercentPoints]) > 0:
            print('raw score interval percent is not percent value !\n', self.__rawScorePercentPoints)
            return False
        # claculate _rawScorePoints
        if field in self.rawdf.columns.values:
            __rawdfdesc = self.rawdf[field].describe(self.__rawScorePercentPoints)
            # calculating _rawScorePoints
            self.__rawScorePoints__ = []
            for f in __rawdfdesc.index:
                if '%' in f:
                    self.__rawScorePoints__ += [__rawdfdesc.loc[f]]
            # deplicated
            # for x in self.__rawScorePercentPoints:
            #    fname = '{0}%'.format(int(x * 100))
            #    self.__rawScorePoints__ += [__rawdfdesc.loc[fname]]
        else:
            print('error score field name!')
            print('not in ' + self.rawdf.columns.values)
            return False
        # calculate Coefficients
        return self.__getcoeff()

    def __pltrun(self, scorefieldname):
        if not self.__preprocess(scorefieldname):
            print('fail to initializing !')
            return
        # transform score
        self.outdf[scorefieldname + '_plt'] = \
            self.outdf[scorefieldname].apply(self.__getcore)

    def __plotmodel(self):
        plt.figure('Piecewise Linear Score Transform: {0}'.format(self.scorefields))
        plen = len(self.__rawScorePoints__)
        plt.xlim(self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1])
        plt.ylim(self.__stdScorePoints[0], self.__stdScorePoints[plen-1])
        plt.plot(self.__rawScorePoints__, self.__stdScorePoints)
        plt.plot([self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]],
                 [self.__rawScorePoints__[0], self.__rawScorePoints__[plen - 1]],
                 )
        plt.xlabel('piecewise linear transform model')
        plt.show()
        return


class Zscore(ScoreTransformModel):
    __doc__ = '''
    transform raw score to Z-score according to percent position on normal cdf
    input data: 
    rawdf = raw score dataframe
    stdNum = standard error numbers
    output data:
    outdf = result score with raw score field name + '_z'
    '''
    HighPrecise = 0.9999999
    MinError = 0.1 ** 9

    def __init__(self):
        super(Zscore, self).__init__()
        self.modelname = 'z'
        self.stdNum = 3
        self.maxRawscore = 150
        self.minRawscore = 0
        self._segtable = None
        self.__currentfield = None
        # create norm table
        self._samplesize = 100000    # cdf error is less than 0.0001
        self._normtable = pl.create_normaltable(self._samplesize, stdnum=4)
        self._normtable.loc[max(self._normtable.index), 'cdf'] = 1

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, stdnum=3, maxscore=100, minscore=0):
        self.stdNum = stdnum
        self.maxRawscore = maxscore
        self.minRawscore = minscore

    def check_parameter(self):
        if self.maxRawscore <= self.minRawscore:
            print('max raw score or min raw score error!')
            return False
        if self.stdNum <= 0:
            print('std number is error!')
            return False
        return True

    def run(self):
        # check data and parameter in super
        if not super().run():
            return
        self.outdf = self.rawdf[self.scorefields]
        self._segtable = self._getsegtable(self.outdf, self.maxRawscore, self.minRawscore, self.scorefields)
        for sf in self.scorefields:
            print('start run...')
            st = time.clock()
            self._calczscoretable(sf)
            # print(f'zscoretable finished with {time.clock()-st} consumed')
            self.outdf[sf+'_zscore'] = self.outdf[sf].\
                apply(lambda x: x if x in self._segtable.seg.values else np.NaN)
            self.outdf[sf+'_zscore'] = \
                self.outdf[sf+'_zscore'].replace(self._segtable.seg.values,
                                                 self._segtable[sf+'_zscore'].values)
            print(f'zscore transoform finished with {time.clock()-st} consumed')

    def _calczscoretable(self, sf):
        if sf+'_percent' in self._segtable.columns.values:
            self._segtable[sf+'_zscore'] = self._segtable[sf+'_percent'].apply(self.__get_zscore_from_normtable)
        else:
            print(f'error: not found field{sf+"_percent"}!')

    def __get_zscore_from_normtable(self, p):
        df = self._normtable.loc[self._normtable.cdf >= p - Zscore.MinError][['sv']].head(1).sv
        y = df.values[0] if len(df) > 0 else None
        if y is None:
            print(f'error: cdf value[{p}] can not find zscore in normtable!')
            return y
        return max(-self.stdNum, min(y, self.stdNum))

    def _getsegtable(self, df, maxscore, minscore, scorefieldnamelist):
        """no sort problem in this segtable usage"""
        seg = pl.SegTable()
        seg.set_data(df, scorefieldnamelist)
        seg.set_parameters(segmax=maxscore, segmin=minscore)
        # seg.rawdf = df
        # seg.segmax = maxscore
        # seg.segmin = minscore
        # seg.segfields = scorefieldnamelist
        seg.run()
        return seg.segdf

    def report(self):
        if type(self.outdf) == pd.DataFrame:
            print('output score desc:\n', self.outdf.describe())
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('parameters:\n')
        print(f'\tzscore stadard diff numbers:{self.stdNum}')
        print(f'\tmax score in raw score:{self.maxRawscore}')
        print(f'\tmin score in raw score:{self.minRawscore}')

    def plot(self, mode='out'):
        if mode in 'raw,out':
            super().plot(mode)
        else:
            print('not support this mode!')


class Tscore(ScoreTransformModel):
    __doc__ = '''
    T分数是一种标准分常模，平均数为50，标准差为10的分数。
    即这一词最早由麦柯尔于1939年提出，是为了纪念推孟和桑代克
    对智力测验，尤其是提出智商这一概念所作出的巨大贡献。'''

    def __init__(self):
        super().__init__()
        self.modelname = 't'
        self.rscore_max = 150
        self.rscore_min = 0
        self.tscore_std = 10
        self.tscore_mean = 50
        self.tscore_max = 150
        self.tscore_min = 0
        self._tscore_stdnum = 3

    def set_data(self, rawdf=None, scorefields=None):
        self.rawdf = rawdf
        self.scorefields = scorefields

    def set_parameters(self, tscore_mean=50, tscore_std=10, tscore_max=100, tscore_min=0):
        self.tscore_mean = tscore_mean
        self.tscore_std = tscore_std
        self.tscore_max = tscore_max
        self.tscore_min = tscore_min
        self._tscore_stdnum = (self.tscore_mean - self.tscore_min) / self.tscore_std

    def run(self):
        zm = Zscore()
        zm.set_data(self.rawdf, self.scorefields)
        zm.set_parameters(stdnum=self._tscore_stdnum, minscore=self.rscore_min,
                          maxscore=self.rscore_max)
        zm.run()
        self.outdf = zm.outdf
        for sf in self.outdf.columns:
            self.outdf[sf].apply(lambda x: (x + self.tscore_mean) * self.tscore_std)

    def report(self):
        if type(self.rawdf) == pd.DataFrame:
            print('raw score desc:\n', pl.report_stats_describe(self.rawdf))
        else:
            print('output score data is not ready!')
        if type(self.outdf) == pd.DataFrame:
            print('output score desc:\n', pl.report_stats_describe(self.outdf))
        else:
            print('output score data is not ready!')
        print(f'data fields in rawscore:{self.scorefields}')
        print('parameters:')
        print(f'\tzscore stadard deviation numbers:{self.tscore_std}')
        print(f'\tmax score in raw score:{self.rscore_max}')
        print(f'\tmin score in raw score:{self.rscore_min}')

    def plot(self, mode='raw'):
        super().plot(mode)


