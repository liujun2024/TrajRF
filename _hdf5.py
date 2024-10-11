""" This package is for reading and writing hdf5 files """

import os
import h5py
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import _era5 as era5


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


def data2hdf(file: h5py.File, location: str, data: pd.DataFrame | dict, attrs=None, **kwargs):
    """ # 将数据存入hdf5文件

        file: h5py.File()
        location: 保存在hdf5文件中的相对位置
        data: 待写入的数据
        attr: dict | None 待写入的属性

    无返回值
    2023-06-25 v1
    单进程
    """

    # 判断data数据类型
    if attrs is None:
        attrs = dict()

    if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        data_np = data.to_numpy()

        # if data.index.name == 'datetime':
        #     index_ = data.index.to_numpy().astype(np.uint64) / 1E9
        #     # attrs['datetime'] = (data.index.to_numpy().astype(np.uint64) * 1E-9).astype(np.int)
        # else:
        #     # attrs['index'] = data.index.to_numpy().astype(np.int)
        #     index_ = data.index.to_numpy().astype(np.int)

        if isinstance(data, pd.DataFrame):
            attrs['columns'] = data.columns.to_numpy()
        else:
            attrs['name'] = data.name

    elif isinstance(data, np.ndarray):
        data_np = data

    else:
        raise ValueError('未识别的Index')

    # # 写入index
    # location_index = location.split('/')[0] + '/' + data.index.name
    # print(location_index)
    # file.create_dataset(name=location_index, data=index_, shuffle='T', compression='gzip', compression_opts=5)

    # 写入数组数据
    file.create_dataset(name=location, data=data_np, shuffle='T', compression='gzip', compression_opts=5)

    # 写入属性数据
    for key in attrs.keys():
        file[location].attrs[key] = attrs[key]


def index2hdf(file: h5py.File, location: str, index):
    """ # 将索引数据存入hdf5文件

        file: h5py.File()
        location: 保存在hdf5文件中的相对位置
        index: 待写入的数据

    无返回值
    2023-06-25 v1
    2023-07-19 v1.1 适配非时间索引数据
    单进程
    """

    # 判断data数据类型，如果是时间索引pd.DatetimeIndex，则将之转换为时间戳（s）
    if isinstance(index, pd.DatetimeIndex):
        index_ = index.to_numpy().astype(np.uint64) * 1E-9
        # name = 'datetime'

    else:
        index_ = index.to_numpy()
        # name = 'index'

    if location not in file:
        # 写入index
        file.create_dataset(name=location, data=index_, shuffle='T', compression='gzip', compression_opts=5)
        file[location].attrs['name'] = index.name

    else:
        raise KeyError('已存在的dataset：%s' % location)


def train2hdf(path_hdf5: str, data: pd.DataFrame):
    """ 将训练数据存入hdf5文件

        path_hdf5: os.PathLise，hdf5文件目标路径
        data: pd.DataFrame，待训练数据，含有datetime索引，最后一列为因变量，其它列为自变量

    无返回值
    2023-06-20 v1
    2023-07-19 v1.1 适配非时间索引数据
    单进程
    """

    # 新建h5文件，存在则替换
    # f = h5py.File(name=path_hdf5, mode='w')
    f = h5py.File(name=path_hdf5, mode='a')

    # 写入数据
    data2hdf(file=f, location='Raw/Train', data=data)
    # if location not in f:
    #     data2hdf(file=f, location=location, data=data)

    # 写入index
    index2hdf(file=f, location='Index/Train', index=data.index)

    # if data.index.name == 'datetime':
    #     index2hdf(file=f, location='Index/Train', index=data.index)

    # 关闭文件
    f.close()


def hdf2train(path_hdf5: str):
    """ 读取训练数据

        path_hdf5: os.PathLise，hdf5文件路径
        location: 保存在hdf5文件中的相对位置

        return: pd.DataFrame，待训练的数据

    2023-06-21 v1
    2023-07-19 v1.1 适配非时间索引数据
    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='r')

    # 读取数据
    data = f['Raw/Train']

    # 自变量、因变量表头
    columns = data.attrs['columns']

    # index
    index = f['Index/Train']

    # index_name
    index_name = index.attrs['name']

    # 将index（时间戳）转换为时间
    if index_name == 'datetime':
        index = pd.to_datetime(index, unit='s')

    # 创建pd.DataFrame
    df_data = pd.DataFrame(data=data, index=index, columns=columns)

    # 设置索引列列名
    df_data.index.name = index_name

    # 关闭文件
    f.close()

    # 返回数据
    return df_data


def test2hdf(path_hdf5: str, data: pd.DataFrame):
    """ 将用于验证的数据存入hdf5文件

        path_hdf5: os.PathLise，hdf5文件目标路径
        data: pd.DataFrame，验证数据，含有datetime索引，最后一列为因变量，其它列为自变量

    无返回值
    2023-06-20 v1
    2023-07-19 v1.1 适配非时间索引数据
    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='a')

    # 写入数据
    data2hdf(file=f, location='Raw/Test', data=data)
    # if location not in f:
    #     data2hdf(file=f, location=location, data=data)

    # 写入index
    index2hdf(file=f, location='Index/Test', index=data.index)

    # if data.index.name == 'datetime':
    #     index2hdf(file=f, location='Index/Test', index=data.index)

    # 关闭文件
    f.close()


def hdf2test(path_hdf5: str):
    """ 读取验证数据

        path_hdf5: os.PathLise，hdf5文件路径
        location: 保存在hdf5文件中的相对位置

        return: pd.DataFrame，待训练的数据

    2023-06-21 v1
    2023-07-19 v1.1 适配非时间索引数据

    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='r')

    # 判断数据是否存在
    if 'Raw/Test' not in f:
        f.close()
        return None

    # 读取数据
    data = f['Raw/Test']

    # 自变量、因变量表头
    columns = data.attrs['columns']

    # index
    index = f['Index/Test']

    # index_name
    index_name = index.attrs['name']

    # 将index（时间戳）转换为时间
    if index_name == 'datetime':
        index = pd.to_datetime(index, unit='s')

    # 创建pd.DataFrame
    df_data = pd.DataFrame(data=data, index=index, columns=columns)

    # 设置索引列列名
    df_data.index.name = index_name

    # 关闭文件
    f.close()

    # 返回数据
    return df_data


def lc2hdf(path_hdf5: str, dict_all_params: dict, dict_best_param: dict, location: str):
    """ 将学习曲线调参过程的数据存入hdf5文件

        path_hdf5: os.PathLise，hdf5文件目标路径
        dict_all_params: dict，过程中每一步的参数的组成的字典，如：
                {
                'n_estimators': pd.Series（index为参数值，data为对应的模型表现）,
                'max_depth': 同上,
                'min_samples_split': 同上,
                'min_samples_leaf': 同上,
                'max_features': 同上,
                'max_samples': 同上,
                 }

        dict_best_param: dict，6个参数的最优值，如：
                {
                'n_estimators': 59,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 2,
                'max_features': 0.5,
                'max_samples': 0.7,
                 }

        location: 保存在hdf5文件中的相对位置

    无返回值
    2023-06-20 v1
    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='a')

    # 如果数据已存在，则删除
    if location in f:
        del f[location]

    # 写入数据
    for p in dict_all_params.keys():
        # hdf5文件中的位置
        location_p = location + p

        # 属性数据
        dict_attrs_p = {
                'index': dict_all_params[p].index,
                # 'data': dict_all_params[p].to_numpy(),
                'name': p,
                'optimal': dict_best_param[p],
        }

        # 写入数据
        data2hdf(file=f, location=location_p, data=dict_all_params[p].to_numpy(), attrs=dict_attrs_p)

    # 保存完关闭文件
    f.close()


def hdf2lc(path_hdf5: str, location: str):
    """ 读取模型训练学习曲线结果，与lc2hdf函数互逆

        path_hdf5: os.PathLise，hdf5文件路径
        location: 保存在hdf5文件中的相对位置

        return: tuple，(dict，dict) 分别为dict_all_params和dict_best_param

    2023-06-21 v1
    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='r')

    # 参数列表
    list_lc = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_samples']

    # 用于存储所有参数
    dict_all_params = dict()

    # 用于存储最优参数
    dict_optimal_param = dict()

    # 读取
    for p in list_lc:
        # 位置
        location_p = location + p

        # 所有参数
        series_p = pd.Series(data=f[location_p], index=f[location_p].attrs['index'], name=f[location_p].attrs['name'])
        dict_all_params[p] = series_p

        # 最优参数
        dict_optimal_param[p] = f[location_p].attrs['optimal']

    # 关闭文件
    f.close()

    # 返回数据
    return dict_all_params, dict_optimal_param


def predict2hdf(path_hdf5: str, dict_predict: dict, location: str):
    """ 将评估结果存入hdf5文件

        path_hdf5: os.PathLise，hdf5文件目标路径
        dict_predict: dict，评估结果
            {
            'r2': float,
            'rmse': float,
            'data': pd.Series,
            }

        location: 保存在hdf5文件中的相对位置

    无返回值
    2023-06-21
    单进程

    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='a')

    # 如果数据已存在，则删除
    if location in f:
        del f[location]

    # 属性数据
    dict_attrs = {
            'r2': dict_predict['r2'],
            'rmse': dict_predict['rmse'],
    }

    # 写入数据
    data2hdf(file=f, location=location, data=dict_predict['data'], attrs=dict_attrs)

    # 关闭文件
    f.close()


def hdf2predict(path_hdf5: str, location: str):
    """ 读取预测数据，与predict2hdf函数互逆

        path_hdf5: os.PathLise，hdf5文件路径
        location: 保存在hdf5文件中的相对位置

        return: dict，预测结果
            {
            'r2': float,
            'rmse': float,
            'data': pd.Series,
            }

    2023-06-25 v1
    单进程
    """

    # 打开h5文件
    f = h5py.File(name=path_hdf5, mode='r')

    # 判断数据是否存在
    if location not in f:
        f.close()
        return None

    # 读取数据
    data = f[location]

    # index位置
    location_index = location.replace('Predict', 'Index')

    # index
    index = f[location_index]

    # index_name
    index_name = index.attrs['name']

    # index
    if index_name == 'datetime':
        index = pd.to_datetime(index, unit='s')

    # 生成pd.Series
    series_data = pd.Series(
        data=data,
        index=index,
        name=index_name
    )

    # 准备返回数据
    dict_data = {
            'r2': data.attrs['r2'],
            'rmse': data.attrs['rmse'],
            'data': series_data,
    }

    # 关闭文件
    f.close()

    # 返回数据
    return dict_data


def shap2hdf(dict_shap: dict, path_hdf5: os.PathLike, location: str):
    """ 将由cal_shap_xx计算的结果存入hdf5文件

        dict_shap: dict, 如：
            {'shap_values_df': pd.DataFrame,
             'shap_expected_value': float,
             'global_shap_df': pd.Series,
            }

        path_hdf5: os.PathLike，保存路径
        location: str，保存在hdf5文件中的相对路径

    无返回值
    2023-06-19 v1
    单进程
    """

    # 打开hdf5文件
    f = h5py.File(name=path_hdf5, mode='a')

    # 判断是否已经存在，是则删除
    if location in f:
        del f[location]

    # 属性数据
    dict_attrs = {
            'shap_expected_value': dict_shap['shap_expected_value'],
    }

    # 写入数据
    data2hdf(file=f, location=location, data=dict_shap['shap_values_df'], attrs=dict_attrs)

    # 关闭文件
    f.close()


def hdf2shap(path_hdf5: str, location: str):
    """ 从hdf5文件中读取shapley value

        dict_shap: dict, 如：
            {'shap_values_df': pd.DataFrame,
             'shap_expected_value': float,
             'global_shap_df': pd.Series,
            }

        path_hdf5: os.PathLike，保存路径
        location: str，保存在hdf5文件中的相对路径

        return: dict, {
            'shap_values_df': pd.DataFrame，auto_shap.generate_shap_values计算的shap_values_df
            'shap_expected_value':  float, shapley value的期望值
        }

    2023-06-19 v1
    单进程
    """

    # 打开hdf5文件
    f = h5py.File(name=path_hdf5, mode='r')

    # 读取数据
    data = f[location]

    # 生成pd.DataFrame
    df_data = pd.DataFrame(
        data=data,
        index=pd.to_datetime(f['Index/Train'], unit='s'),
        # index=pd.to_datetime(f['Index/Training'], unit='s'),
        # index=pd.to_datetime(data.attrs['datetime'], unit='s'),
        columns=data.attrs['columns'],
    )
    df_data.index.name = 'datetime'

    # 准备返回数据
    dict_data = {
            'shap_values_df': df_data,
            'shap_expected_value': data.attrs['shap_expected_value'],
    }

    # 关闭文件
    f.close()

    # 返回数据
    return dict_data


if __name__ == '__main__':
    pass
