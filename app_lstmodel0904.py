import pandas as pd
import py2ee_ssm0906 as pm
import numpy as np


def lstmodel_gkdata(rawdf, kms='1'):
    lst = pm.PltScoreModel()
    if kms == '1':
        df = rawdf[(rawdf.km1 > 0) & (rawdf.kldm.isin(list('13457')))][['km1']]
        kms = 'km1'
    elif kms == '2w':
        df = rawdf[(rawdf.km2 > 0) & (rawdf.kldm.isin(list('13')))][['km2']]
        kms = 'km2'
    elif kms == '2l':
        df = rawdf[(rawdf.km2 > 0) & (rawdf.kldm.isin(list('457')))][['km2']]
        kms = 'km2'
    elif kms == '3':
        df = rawdf[(rawdf.km3 > 0) & (rawdf.kldm.isin(list('13457')))][['km3']]
        kms = 'km3'
    else:
        print('kms is error!')
        return
    # [0, .15, .30, .50, .70, .85, 1.00]    # adjust much
    # rawpoints = [0, 0.02, 0.15, 0.50, 0.85, 0.98, 1]   # little adjust normal ratio
    rawpoints = [0, 0.0228, 0.1587, 0.50, 0.8414, 0.9773, 1]   # normal ratio
    # stdpoints = [50, 60, 80, 100, 120, 140, 150]  # [0, 150] => [50, 150], std=20
    stdpoints = [50, 70, 85, 100, 115, 130, 150]  # [0, 150] => [50, 150], std=15
    lst.set_parameters(rawpoints, stdpoints)
    lst.set_data(df, [kms])
    lst.run()
    lst.report()
    lst.plot('raw')
    lst.plot('out')
    lst.plot('model')
    return


def get_gkdata(year):
    rawdf = pd.read_csv(r'test\{0}cj.csv'.format(year),
                        dtype={'KM1': np.int16, 'KM2': np.int16, 'KM3': np.int16,
                               'KM4': np.int16, 'KLDM': np.str}
                        )
    rawdf = rawdf.rename(columns={'KM1': 'km1', 'KM2': 'km2', 'KM3': 'km3', 'KM4': 'km4', 'KLDM': 'kldm'})
    return rawdf
