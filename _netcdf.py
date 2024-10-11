""" This package is for reading *.nc files of ERA5 """

import pandas as pd
import xarray as xr


def read_era5_without_levels_by_coord(list_files: str, coord: pd.DataFrame):
    """ 从ERA5 land或ERA5 single levels数据(无level维度)中读取多个坐标点的时间序列

        list_files: ERA5 land或ERA5 single levels数据路径列表

        coord: pd.DataFrame, index为site code, 并含有lat和lon列

        return: dict, 如：{
                           'short_name': 't2m',
                           'data': pd.DataFrame(),  # 时间序列, 列名为site code
                          }

    2023-09-16 v1
    单进程单线程
    """

    # 坐标xr.dataArray
    index_lat = coord['lat'].to_xarray()
    index_lon = coord['lon'].to_xarray()
    # print('index_lat:\n', index_lat)

    """ 利用xarray读取数据(决速步骤1) """
    # ds_ori = xr.open_mfdataset(paths=list_files, engine='scipy')
    ds_ori = xr.open_mfdataset(paths=list_files, engine='netcdf4')

    # 经纬度筛选
    if 'expver' in ds_ori.coords:
        ds_sel = ds_ori.sel(latitude=index_lat, longitude=index_lon, method='nearest', expver=1)
    else:
        ds_sel = ds_ori.sel(latitude=index_lat, longitude=index_lon, method='nearest')

    # print('ds_sel:\n', ds_sel)

    # 变量名
    short_name = [i for i in ds_sel.data_vars][0]
    long_name = ds_sel[short_name].attrs['long_name']

    # 单位
    units = ds_sel[short_name].attrs['units']

    # 转换为pd.Series（决速步骤2）
    _series = ds_sel[short_name].to_series()
    _df = _series.unstack(level=1)
    _df.index.name = 'datetime'
    # print(_df.columns.tolist())

    """ 还原ERA5中的累计数据(如tp、str、strd、ssr、ssrd...)为小时值，
        以降雨量为例，转换降雨量数据累计值为小时值

        ERA5_Land_Hourly中的小时降雨量数据实际上是累积值,
            从第1天1:00开始, 第2天0:00结束, 1:00代表的是实际降水量, 2:00代表的是1:00至2:00的累计降水量,
            以此类推, 第2天0:00代表的是第1天的日降水量 
    """

    if short_name in cfg.list_var_accumulations:
        # 提取1:00的数据
        df_hour1 = _df[_df.index.hour == 1]

        # 按照时间轴的方向，后面的数据减去前面的数据，第1个时刻的数据摒弃
        _df = _df.diff(periods=1, axis=0)

        # 因为1:00的数据不用减去前面时刻的值，因此将前面提取出来的1:00时刻的值还原
        _df[_df.index.hour == 1] = df_hour1

    """ 删除nan行 """
    _df.dropna(how='all', axis=0, inplace=True)

    # 返回数据
    return {'data': _df, 'short_name': short_name, 'long_name': long_name, 'units': units}


# def read_era5_with_levels_by_coord(dir_data: os.PathLike, coord: pd.DataFrame, utc_offset=8, levels=[850, 1000]):
#     """ 从ERA5 pressure levels数据（含有level维度）中读取多个坐标点的时间序列

#         dir_data: ERA5 pressure levels数据某变量的存放路径

#         coord: pd.DataFrame，index为site code，并含有lat和lon列

#         utc_offset：时区，默认为=8，即北京时间

#         level: list，需要读取的pressure levels

#         return: dict, 如：{
#                            'short_name': 'u',
#                            'data': [pd.DataFrame(), ...],  # 时间序列，列名为site code
#                           }

#     2023-05-21 v1
#     单进程单线程
#     """

#     # 文件列表
#     list_files = [os.path.join(dir_data, i) for i in os.listdir(dir_data) if i.endswith('.nc')]

#     # 坐标xr.dataArray
#     index_lat = coord['lat'].to_xarray()
#     index_lon = coord['lon'].to_xarray()
#     # print('index_lat:\n', index_lat)

#     """ 利用dask进行处理（决速步骤1） """
#     ds_ori = xr.open_mfdataset(paths=list_files, engine='netcdf4')
#     # print('_ds:\n', _ds_ori)

#     # 变量名
#     short_name = [i for i in ds_ori.data_vars][0]

#     # 经纬度及level筛选
#     list_df = []
#     for level in levels:
#         ds_level = ds_ori.sel(latitude=index_lat, longitude=index_lon, level=level, method='nearest')

#         # 转换为pd.Series（决速步骤2）
#         series_level = ds_level[short_name].to_series()
#         df_level = series_level.unstack(level=1)
#         df_level.index.name = 'datetime'
#         # print(_df)

#         """ 删除nan行 """
#         df_level.dropna(how='all', axis=0, inplace=True)

#         """ 时间校正, UTC -> UTC+8 (北京时间) """
#         df_level.index = df_level.index + pd.Timedelta(hours=utc_offset)

#         """ 单位转换
#             t: K --> °C;
#         """

#         # # 温度
#         # if short_name in ['t']:
#         #     df_level = df_level - 273.15

#         # else:
#         #     pass

#         print('\n\n', df_level)
#         list_df.append(df_level)

#     # 返回数据
#     return {'data': list_df, 'short_name': short_name}

