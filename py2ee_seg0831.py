# -*- utf-8 -*-
import pandas as pd
import numpy as np


# example for SegTable
def exp_segtable():
    df = pd.DataFrame({'a': [x % 100 for x in range(10000)],
                       'b': [min(max(int(np.random.randn()*10+50), 0), 100) for x in range(10000)]})
    seg = SegTable()
    seg.inputdf = df
    seg.segsort = 'desending'
    seg.segstep = 5
    seg.segmax = 99
    seg.segmin = 0
    seg.segfields = ['a', 'b']
    seg.run()
    return seg.outputdf


# segment table for score dataframe
class SegTable(object):
    __doc__ = '''
    input: dataframe with a value field(int,float) to calculate segment table
    parameters:
        segStep: int, levels for segment value
        segMax: int,  maxvalue for segment
        segMin: int, minvalue for segment
        segFields: list, field names to calculate, empty for calculate all fields
    output:dataframe with field 'seg, segfield_count, segfield_cumsum, segfield_percent'
    example:
        seg = ps.SegTable()
        seg.inputdf = pd.DataFrame({'sf':[i for i in range(1000)]})
        seg.segStep = 1     #default:1
        seg.segMax = 100    #default:150
        seg.segMin = 1      #default:0
        seg.segSort = 'desencding'  #default:'ascending'
        seg.run()
        seg.outputdf    #fields:sf, sf_count, sf_cumsum, sf_percent
    Note:
        score value >=0
        score fields type is int
    '''

    def __init__(self):
        self.__df = None
        self.__segdf = None
        self.__segStep = 1
        self.__segMax = 150
        self.__segMin = 0
        self.__segSort = 'ascending'
        self.__segFields = []
        return

    @property
    def segfields(self):
        return self.__segFields

    @segfields.setter
    def segfields(self, sf):
        self.__segFields = sf
        return

    @property
    def segstep(self):
        return self.__segStep

    @segstep.setter
    def segstep(self, segstep):
        self.__segStep = segstep
        return

    @property
    def segmin(self):
        return self.__segMin

    @segmin.setter
    def segmin(self, smin):
        self.__segMin = smin
        return

    @property
    def segmax(self):
        return self.__segMax

    @segmax.setter
    def segmax(self, smax):
        self.__segMax = smax

    @property
    def segsort(self):
        return self.__segSort

    @segsort.setter
    def segsort(self, sort):
        self.__segSort = sort
        return

    @property
    def rawdf(self):
        return self.__df

    @rawdf.setter
    def inputdf(self, df):
        self.__df = df
        return

    @property
    def outdf(self):
        return self.__segdf

    def run(self):
        if type(self.__df) == pd.Series:
            self.__df = pd.DataFrame(self.__df)
        if type(self.__df) != pd.DataFrame:
            print('data set is not ready!')
            return
        # create output dataframe with segstep = 1
        seglist = [x for x in range(self.__segMin, self.__segMax + 1)]
        self.__segdf = pd.DataFrame({'seg': seglist})
        if not self.__segFields:
            self.__segFields = self.__df.columns.values
        for f in self.__segFields:
            r = self.__df.groupby(f)[f].count()
            self.__segdf[f+'_count'] = self.__segdf['seg'].apply(lambda x: r[x] if x in r.index else 0)
            if self.__segSort != 'ascending':
                self.__segdf = self.__segdf.sort_values(by='seg', ascending=False)
            self.__segdf[f+'_cumsum'] = self.__segdf[f+'_count'].cumsum()
            maxsum = max(max(self.__segdf[f+'_cumsum']), 1)
            self.__segdf[f+'_percent'] = self.__segdf[f+'_cumsum'].apply(lambda x: x/maxsum)
            if self.__segStep > 1:
                segcountname = f+'_count{0}'.format(self.__segStep)
                segcumsumname = f+'_cumsum{0}'.format(self.__segStep)
                segpercentname = f+'_percent{0}'.format(self.__segStep)
                self.__segdf[segcountname] = -1
                c = 0
                if self.__segSort == 'ascending':
                    curpoint = self.__segMin
                    curstep = self.__segStep
                else:
                    curpoint = self.__segMax
                    curstep = -self.__segStep
                for x in self.__segdf['seg']:
                    c += self.__segdf.loc[self.__segdf['seg'] == x, f+'_count'].values[0]
                    if x == curpoint:
                        self.__segdf.loc[self.__segdf.seg == x, segcountname] = c
                        c = 0
                        curpoint += curstep
                self.__segdf[segcumsumname] = self.__segdf[segcountname][self.__segdf[segcountname] >= 0].cumsum()
                self.__segdf[self.__segdf.isnull()] = -1
                self.__segdf[segpercentname] = self.__segdf[segcumsumname].\
                    apply(lambda x: x / max(self.__segdf[segcumsumname]) if x > 0 else x)
        return
# SegTable class end
