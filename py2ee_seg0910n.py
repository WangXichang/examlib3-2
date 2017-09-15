# -*- utf-8 -*-
import pandas as pd
import numpy as np


# example for SegTable
def exp_segtable():
    import time
    df = pd.DataFrame({'a': [np.int(time.clock()*1000000) % 101 for x in range(1000000)],
                       'b': [min(max(int(np.random.randn()*10+50), 0), 100) for x in range(1000000)]})
    seg = SegTable()
    seg.set_data(df, 'a')
    seg.set_parameters(segmax=100, segmin=0, segstep=5, segsort='ascending')
    st = time.clock()
    seg.run()
    print(time.clock()-st)
    return seg, df


# segment table for score dataframe
class SegTable(object):
    __doc__ = '''
    input: rawdf, dataframe, with a value field(int,float) to calculate segment table
           segfields, list, field names to calculate, empty for calculate all fields
    parameters:
        segstep: int, levels for segment value
        segmax: int,  maxvalue for segment
        segmin: int, minvalue for segment
    output:dataframe with field 'seg, segfield_count, segfield_cumsum, segfield_percent'
    example:
        seg = ps.SegTable()
        df = pd.DataFrame({'sf':[i for i in range(1000)]})
        seg.set_data(df, 'sf')
        seg.set_parameters(segmax=100, segmin=1, segstep=1, segsort='descending')
        seg.run()
        seg.segdf    #result dataframe, with fields: sf, sf_count, sf_cumsum, sf_percent
    Note:
        score value >=0
        score fields type is int
    '''

    def __init__(self):
        self.__rawDf = None
        self.__segFields = []
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'ascending'
        self.__segDf = None
        return

    @property
    def segdf(self):
        return self.__segDf

    @segdf.setter
    def segdf(self, df):
        self.__segDf = df

    @property
    def rawdf(self):
        return self.__rawDf

    @property
    def segfields(self):
        return self.__segFields

    @segfields.setter
    def segfields(self, sfs):
        self.__segFields = sfs

    def set_data(self, df, segfields):
        self.__rawDf = df
        self.__segFields = segfields

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
        return True

    def run(self):
        if not self.check():
            return
        # create output dataframe with segstep = 1
        seglist = [x for x in range(self.__segMin, self.__segMax + 1)]
        self.segdf = pd.DataFrame({'seg': seglist})
        if not self.segfields:
            self.segfields = self.rawdf.columns.values
        for f in self.segfields:
            r = self.rawdf.groupby(f)[f].count()
            self.segdf[f + '_count'] = self.segdf['seg'].\
                apply(lambda x: np.int64(r[x]) if x in r.index else 0)
            if self.__segSort != 'ascending':
                self.segdf = self.segdf.sort_values(by='seg', ascending=False)
            self.segdf[f + '_cumsum'] = self.__segDf[f + '_count'].cumsum()
            maxsum = max(max(self.segdf[f + '_cumsum']), 1)
            self.segdf[f + '_percent'] = self.segdf[f + '_cumsum'].apply(lambda x: x / maxsum)
            if self.__segStep > 1:
                segcountname = f+'_count{0}'.format(self.__segStep)
                # segcumsumname = f+'_cumsum{0}'.format(self.__segStep)
                # segpercentname = f+'_percent{0}'.format(self.__segStep)
                self.segdf[segcountname] = np.int64(-1)
                # self.segdf[segcumsumname] = np.NaN
                # self.segdf[segpercentname] = np.NaN
                c = 0
                curpoint, curstep = ((self.__segMin, self.__segStep)
                                     if self.__segSort == 'ascending' else
                                     (self.__segMax, -self.__segStep))
                for index, row in self.segdf.iterrows():
                    c += row[f+'_count']
                    if np.int64(row['seg']) in [curpoint, self.__segMax, self.__segMin]:
                        row[segcountname] = c
                        # self.segdf.loc[index, [segcountname, segcumsumname, segpercentname]] = \
                        #    c, row[f+'_cumsum'], row[f+'_percent']
                        # self.segdf.loc[index, segcountname] = np.int64(c)
                        c = 0
                        curpoint += curstep
        return
# SegTable class end
