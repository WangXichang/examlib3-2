# -*- utf-8 -*-
import pandas as pd
import scipy.stats as stats
import py2ee_lib as pl

for year in [2015, 2016, 2017]:
    gkdf = pd.read_csv(fr'test\{year}cj.csv')
    print(f'{year} data analysis:\n','='*50)
    print('(yuwen, waiyu)\n', '-'*50)
    pl.report_stats_describe(gkdf.loc[gkdf.KLDM.isin(list('13457')),['KM1', 'KM3']])
    print('wenke(1,3)(shuxue, zonghe)\n', '-'*50)
    pl.report_stats_describe(gkdf.loc[gkdf.KLDM.isin(list('13')),['KM2', 'KM4']])
    print('like(4,5,7)(shuxue, zonghe)\n', '-'*50)
    pl.report_stats_describe(gkdf.loc[gkdf.KLDM.isin(list('457')),['KM2', 'KM4']])
    print('='*50,'\n')
