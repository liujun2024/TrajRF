import traceback
import typing
import pandas as pd
import numpy as np
import h5py
import os
import re
import time
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from itertools import chain
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from PyQt6.QtCore import QObject, QThread, pyqtSignal, QCoreApplication
# from functools import partial
# from PyQt6.QtWidgets import QApplication
# from pprint import pprint
# from pyexcelerate import Workbook

import _era5 as era5
# import _config as cfg
# import _cal as cal
# import _plot_html as plot_html


def get_dt_in_hdf(path_hdf: str):
     
    """ 读取hdf5文件中已有数据的时间序列 
        path_hdf: hdf5文件路径

    2023-09-17 v1
    """
    
    # 打开hdf5文件
    f = h5py.File(path_hdf, 'r')

    # 包含的年份列表
    list_year = list(f.keys())

    # 读取每个年份dataset包含的时间序列
    dt_index = pd.to_datetime(np.hstack(tup=[f[year].attrs['datetime'] for year in list_year]), unit='s')

    return dt_index


def read_hdf5_by_file(path_h5: str, date_range: list, sites=None, utc_offset=8):
    """ 读取ERA5提取出的站点hdf5数据: 单物种多个站点
        采用进程池ThreadPoolExecutor处理
        
        path_h5: h5文件路径
        date_range: 日期范围, ['2014-05-13', '2020-10-28']
        sites_: 站点代码列表 ['1001A', '1002A', ...]
        utc_offset: 时区, 默认为=8, 即北京时间

        return: pd.DataFrame

    # 单进程
    2023-09-17 v1
    """

    def read_one_dataset(year_):
        """ 读取其中一个dataset的数据 """

        # 数据集
        dset_ = f[year_]

        # 时间索引
        index_ = dset_.attrs['datetime']

        # hdf5文件中的columns
        columns_ = dset_.attrs['columns']

        # short_name
        short_name_ = dset_.attrs['short_name']

        # sites_与columns的交集
        if sites is None:
            columns_to_read = columns_
        else:
            columns_to_read = np.intersect1d(columns_, sites)

        # 站点代码列表columns_to_read在columns中的位置索引
        columns_index = np.where(np.isin(columns_, columns_to_read))[0]

        # 读取数据，决速步骤，已优化
        array_data = dset_[()][..., columns_index]

        # 单位转换
        array_data = era5.unit_conversion(array=array_data, short_name=short_name_)

        # 生成pd.DataFrame
        df_year = pd.DataFrame(
            data=array_data,
            columns=columns_to_read,
            index=pd.to_datetime(index_, unit='s'),
        )

        return df_year

    # sites：list -> np.1darray
    if sites is not None:
        sites = np.array(sites)

    # 需要读取的原始数据的日期范围（因为nc以及hdf5文件存储的是UTC时间的数据）
    dt_start = (pd.to_datetime(date_range[0]) - pd.Timedelta(hours=utc_offset))
    dt_end = (pd.to_datetime(date_range[1]) - pd.Timedelta(days=-1, hours=utc_offset))

    # 打开hdf5文件
    f = h5py.File(name=path_h5, mode="r", swmr=True)

    # 年列表
    list_year = map(str, list(range(dt_start.year, dt_end.year + 1)))
    list_year = [year for year in list_year if year in f]

    # 开启线程池处理数据
    pool = ThreadPoolExecutor(max_workers=100)

    # 提交任务
    list_df = list(pool.map(read_one_dataset, list_year))

    # 合并不同年数据
    df_data = pd.concat(objs=list_df, axis=0)

    # UTC -> Beijing Time
    df_data.index = df_data.index + pd.Timedelta(hours=utc_offset)

    # 索引名
    df_data.index.name = "datetime"

    # 时间筛选
    df_data = df_data.loc[date_range[0]: date_range[1], ]

    # 返回数据
    return df_data


def read_hdf5_by_sites(dir_h5: str, species: list, date_range: list, sites=None):
    """ 读取ERA5提取出的站点hdf5数据: 多物种多站点
        
        dir_h5: hdf5文件所在目录
        species: 物种名列表, PM2.5、PM10、SO2、NO2、CO、O3、AQI
        sites_: 站点代码列表 ['1001A', '1002A', ...]
        date_range: 日期范围, ['2014-05-13', '2020-10-28']

        return: dict, {'PM2.5': df1, 'PM10': df2, ...}

    进程池ProcessPoolExecutor
    2023-09-17 v1
    """

    # h5文件路径
    dict_path_h5 = {s: os.path.join(dir_h5, s + '.h5') for s in species}

    """ 读取数据-单进程 """
    if len(species) == 1:
        dict_result = {species[0]: read_hdf5_by_file(path_h5=dict_path_h5[species[0]],
                                                    sites=sites,
                                                    date_range=date_range)
                       }

        return dict_result

    """ 读取数据-进程池 """
    # 初始化进程池
    pool = ProcessPoolExecutor(max_workers=10)

    # 准备参数
    args_ = [(dict_path_h5[s], date_range, sites) for s in species]

    # 提交任务
    list_result = list(pool.map(read_hdf5_by_file, *zip(*args_)))

    # 结果
    dict_result = dict(zip(species, list_result))

    # 返回数据
    return dict_result


def df2h5_era5(data_: pd.DataFrame, path_hdf5: str, short_name: str, long_name: str, units: str, mode='a'):
    """ 将读取的ERA5站点数据(pd.DataFrame)保存至数据库(hdf5), 仅包含某年全年的数据

        data_: index为datetime, 各列为每个站点的数据
        path_hdf5: 数据保存目标路径
        short_name: 短变量名    
        long_name: 长变量名    
        units: 单位
    
    无返回值
    2023-08-27 v1
    """

    # 传参
    df_data = data_

    # 删除全是NaN的行
    df_data.dropna(how='all', axis=0, inplace=True)

    # 删除全是NaN的列
    df_data.dropna(how='all', axis=1, inplace=True)
    # print('df_data:\n', df_data.columns.tolist())

    # 按列名排序
    df_data.sort_index(axis=1, inplace=True)
    # print(df_data)

    # 用-999替换NaN
    # df_data.fillna(-999, inplace=True)

    # 数据类型转换，节省存储空间
    df_data = df_data.astype(np.float32)
    # df_data = df_data.astype(np.int16)

    # print('df_data:\n', df_data)

    # 时间索引, 先将DatetimeIndex转换为时间戳
    dt_index = (df_data.index.astype('int64') // 10 ** 9).astype('int32').to_numpy()

    """ 写入hdf5 """
    # 新建hdf5文件，存在则替换
    f = h5py.File(path_hdf5, mode)  # 打开h5文件

    # 数据年份
    year_ = str(df_data.index[0].year)

    if year_ in f.keys():

        # dataset
        ds_ = f[year_]

        # 已存在数据的shape
        shape_existing = ds_.shape

        # 扩展dataset的size
        ds_.resize(shape_existing[0] + data_.shape[0], axis=0)

        # 写入新数据
        ds_[shape_existing[0]:, :] = df_data.to_numpy()

        # 更新datetime
        # dt_index_new = np.hstack((ds_.attrs['datetime'], dt_index))
        # print(dt_index_new, type(dt_index_new))
        ds_.attrs['datetime'] = np.hstack((ds_.attrs['datetime'], dt_index))

    else:
        # 新建dataset并赋值
        # f.create_dataset(year_, data=df_data.to_numpy(), shuffle='T', compression='lzf', maxshape=(None, df_data.shape[1]))
        f.create_dataset(year_, data=df_data.to_numpy(), shuffle='T', compression='gzip', maxshape=(None, df_data.shape[1]), compression_opts=1)

        # 写入属性-表头
        f[year_].attrs['columns'] = df_data.columns.tolist()

        # 写入属性-时间索引
        f[year_].attrs['datetime'] = dt_index

        # 写入属性-变量名
        f[year_].attrs['short_name'] = short_name
        f[year_].attrs['long_name'] = long_name

        # 写入变量-单位
        f[year_].attrs['units'] = units

    # 关闭文件
    f.close()


if __name__ == '__main__':
    pass
