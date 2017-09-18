# -*- utf-8 -*-
# version 2017-09-16

import matplotlib.pyplot as plt
# import matplotlib as mp
import pandas as pd
import numpy as np
import math
# from texttable import Texttable
from scipy import stats


def exp_scoredf_normal(mean=70, std=10, maxscore=100, minscore=0, size=100000):
    """
    生成具有正态分布的分数数据，类型为 pandas.DataFrame, 列名为 sf
    create a score dataframe with fields 'sf', used to test some application
    :parameter
        mean: 均值， std:标准差， maxscore:最大分值， minscore:最小分值， size:人数（样本数）
    :return
        DataFrame, columns = {'sf'}
    """
    df = pd.DataFrame({'sf': [max(minscore, min(int(np.random.randn(1)*std + mean), maxscore))
                              for x in range(size)]})
    return df


# create normal distributed data N(mean,std), [-std*stdNum, std*stdNum], sample points = size
def create_normaltable(size=400, std=1, mean=0, stdnum=4):
    """
    function
        生成正态分布量表
        create normal distributed data(pdf,cdf) with preset std,mean,samples size
        at interval: [-stdNum * std, std * stdNum]
    parameter
        size: samples number for create normal distributed PDF and CDF
        std:  standard difference
        mean: mean value
        stdnum: used to define data range [-std*stdNum, std*stdNum]
    return
        DataFrame: 'sv':stochastic variable, 'pdf':pdf value, 'cdf':cdf value
    """
    interval = [mean - std * stdnum, mean + std * stdnum]
    step = (2 * std * stdnum) / size
    x = [mean + interval[0] + v*step for v in range(size+1)]
    nplist = [1/(math.sqrt(2*math.pi)*std) * math.exp(-(v - mean)**2 / (2 * std**2)) for v in x]
    ndf = pd.DataFrame({'sv': x, 'pdf': nplist})
    ndf['cdf'] = ndf['pdf'].cumsum() * step
    return ndf


# use scipy.stats descibe report dataframe info
def report_stats_describe(dataframe, decdigits=4):
    """
    report statistic describe of a dataframe, with decimal digits = decnum
    峰度（Kurtosis）与偏态（Skewness）就是量测数据正态分布特性的两个指标。
    峰度衡量数据分布的平坦度（flatness）。尾部大的数据分布峰度值较大。正态分布的峰度值为3。
        Kurtosis = 1/N * Sigma(Xi-Xbar)**4 / (1/N * Sigma(Xi-Xbar)**2)**2
    偏态量度对称性。0 是标准对称性正态分布。右（正）偏态表明平均值大于中位数，反之为左（负）偏态。
        Skewness = 1/N * Sigma(Xi-Xbar)**3 / (1/N * Sigma(Xi-Xbar)**2)**3/2
    :param
        dataframe: pandas DataFrame, raw data
        decnum: decimal number in report print
    :return(print)
        records
        min,max
        mean
        variance
        skewness
        kurtosis
    """

    def toround(listvalue, getdecdigits):
        return '  '.join([f'%(v).{getdecdigits}f' % {'v': round(x, getdecdigits)} for x in listvalue])

    def tosqrt(listvalue, getdecdigits):
        return '  '.join([f'%(v).{getdecdigits}f' % {'v': round(math.sqrt(x), getdecdigits)} for x in listvalue])

    # for key, value in stats.describe(dataframe)._asdict().items():
    #    print(key, ':', value)
    sd = stats.describe(dataframe)
    print('\trecords: ', sd.nobs)
    print('\tmin: ', toround(sd.minmax[0], 0))
    print('\tmax: ', toround(sd.minmax[1], 0))
    print('\tmean: ', toround(sd.mean, decdigits))
    print('\tvariance: ', toround(sd.variance, decdigits), '\n\tstdandard deviation: ', tosqrt(sd.variance, decdigits))
    print('\tskewness: ', toround(sd.skewness, decdigits))
    print('\tkurtosis: ', toround(sd.kurtosis, decdigits))
    dict = {'records': sd.nobs, 'max': sd.minmax[1], 'min': sd.minmax[0],
            'mean': sd.mean, 'variance': sd.variance, 'skewness': sd.skewness,
            'kurtosis': sd.kurtosis}
    return dict


# test SegTable
def test_segtable():
    """
    a example for test SegTable
    ------------------------------------------
    expdf = exp_scoredf_normal()
    seg = SegTable()
    seg.set_data(expdf, expdf.columns.values)
    seg.set_parameters(segstep=3)
    seg.run()
    ------------------------------------------
    :return:
        seg.segdf
    """
    expdf = exp_scoredf_normal()
    seg = SegTable()
    seg.set_data(expdf, list(expdf.columns))
    seg.set_parameters(segstep=3)
    seg.run()
    seg.plot()
    return seg


# 计算pandas.DataFrame中分数字段的分段人数表
# segment table for score dataframe
# version 0917-2017
class SegTable(object):
    """
    设置数据，数据表（类型为pandas.DataFrame）,同时需要设置需要计算分数分段人数的字段（list类型）
    :data
        rawdf: input dataframe, with a value fields(int,float) to calculate segment table
        segfields: list, field names used to calculate seg table, empty for calculate all fields
    设置参数
    :parameters
        segmax: int,  maxvalue for segment, default=150
        segmin: int, minvalue for segment, default=0
        segstep: int, levels for segment value, default=1
        segsort: str, 'ascending' or 'descending', default='descending'(sort seg descending)
    运行结果
    :result
        segdf: dataframe with field 'seg, segfield_count, segfield_cumsum, segfield_percent'
    应用举例
    example:
        import py2ee_lib
        seg = py2ee_lib.SegTable()
        df = pd.DataFrame({'sf':[i for i in range(1000)]})
        seg.set_data(df, 'sf')
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='descending')
        seg.run()
        seg.plot()
        resultdf = seg.segdf    # get result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent
    备注
    Note:
        在设定的区间范围内计算分数值，抛弃不再范围内的分数项
        segmax and segmin used to constrain score value scope to be processed in [segmin, segmax]
        分数字段的类型为整数或浮点数（实数）
        score fields type is int or float
    """

    def __init__(self):
        self.__rawDf = None
        self.__segFields = []
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'descending'
        self.__segDf = None
        return

    @property
    def segdf(self):
        return self.__segDf

    @property
    def rawdf(self):
        return self.__rawDf

    @rawdf.setter
    def rawdf(self, df):
        self.__rawDf = df

    @property
    def segfields(self):
        return self.__segFields

    @segfields.setter
    def segfields(self, sfs):
        self.__segFields = sfs

    def set_data(self, df, segfields=''):
        self.rawdf = df
        if (segfields == '') & type(df) == pd.DataFrame:
            self.segfields = df.columns.values
        else:
            self.segfields = segfields

    def set_parameters(self, segmax=100, segmin=0, segstep=1, segsort='descending'):
        self.__segMax = segmax
        self.__segMin = segmin
        self.__segStep = segstep
        self.__segSort = segsort

    def show_parameters(self):
        print('seg max value:{}'.format(self.__segMax))
        print('seg min value:{}'.format(self.__segMin))
        print('seg step value:{}'.format(self.__segStep))
        print('seg sort mode:{}'.format(self.__segSort))

    def check(self):
        if type(self.__rawDf) == pd.Series:
            self.__rawDf = pd.DataFrame(self.__rawDf)
        if type(self.__rawDf) != pd.DataFrame:
            print('data set is not ready!')
            return False
        if self.__segMax <= self.__segMin:
            print('segmax value is not greater than segmin!')
            return False
        if (self.__segStep <= 0) | (self.__segStep > self.__segMax):
            print('segstep is too small or big!')
            return False
        if type(self.segfields) != list:
            if type(self.segfields) == str:
                self.segfields = [self.segfields]
            else:
                print('segfields error:', type(self.segfields))
                return False
            for f in self.segfields:
                if f not in self.rawdf.columns.values:
                    print('field in segfields is not in rawdf:', f)
                    return False
            return True
        return True

    def run(self):
        if not self.check():
            return
        # create output dataframe with segstep = 1
        seglist = [x for x in range(self.__segMin, self.__segMax + 1)]
        self.__segDf = pd.DataFrame({'seg': seglist})
        if not self.segfields:
            self.segfields = self.rawdf.columns.values
        for f in self.segfields:
            r = self.rawdf.groupby(f)[f].count()
            self.segdf[f + '_count'] = self.segdf['seg'].\
                apply(lambda x: np.int64(r[x]) if x in r.index else 0)
            if self.__segSort != 'ascending':
                self.__segDf = self.segdf.sort_values(by='seg', ascending=False)
            self.segdf[f + '_cumsum'] = self.__segDf[f + '_count'].cumsum()
            maxsum = max(max(self.segdf[f + '_cumsum']), 1)
            self.__segDf[f + '_percent'] = self.__segDf[f + '_cumsum'].apply(lambda x: x / maxsum)
            if self.__segStep > 1:
                segcountname = f+'_count{0}'.format(self.__segStep)
                self.__segDf[segcountname] = np.int64(-1)
                c = 0
                curpoint, curstep = ((self.__segMin, self.__segStep)
                                     if self.__segSort == 'ascending' else
                                     (self.__segMax, -self.__segStep))
                for index, row in self.__segDf.iterrows():
                    c += row[f+'_count']
                    if np.int64(row['seg']) in [curpoint, self.__segMax, self.__segMin]:
                        # row[segcountname] = c
                        self.__segDf.loc[index, segcountname] = np.int64(c)
                        c = 0
                        curpoint += curstep
        return

    def plot(self):
        for sf in self.segfields:
            plt.figure('seg table figure')
            plt.subplot(221)
            plt.plot(self.segdf.seg, self.segdf[sf+'_count'])
            plt.title('seg -- count')
            plt.subplot(222)
            plt.plot(self.segdf.seg, self.segdf[sf + '_cumsum'])
            plt.title('seg -- cumsum')
            plt.subplot(223)
            plt.plot(self.segdf.seg, self.segdf[sf + '_percent'])
            plt.title('seg -- percent')
            plt.subplot(224)
            plt.hist(self.rawdf[sf], 20)
            plt.title('raw score histogram')
            plt.show()

# SegTable class end
