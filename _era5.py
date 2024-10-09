from __future__ import annotations
import os
import time
import h5py
import traceback
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from PyQt6.QtCore import QObject, QThread, pyqtSignal, QCoreApplication
# import _cal as cal

import _config as cfg
import _gkd as gkd
import _hdf5 as hdf5


def cal_rh(t, dp, unit='C'):
    """ # 根据温度t和露点温度dp计算相对湿度RH

        t: 摄氏度, 单个值或n维数组
        dp: 摄氏度, 单个值或n维数组
        unit: C (摄氏度)、K(开尔文)

        return: RH, %

    2023-05-22 改为调用MetPy库计算
            或者参考: https://bmcnoldy.earth.miami.edu/Humidity.html

    单进程
    """

    from metpy.calc import relative_humidity_from_dewpoint
    from metpy.units import units

    # 单位
    if unit == 'C':

        _rh = relative_humidity_from_dewpoint(t * units.degC, dp * units.degC).to('percent')

    elif unit == 'K':

        _rh = relative_humidity_from_dewpoint(t * units.kelvin, dp * units.kelvin).to('percent')

    else:
        raise TypeError('单位输入错误！')

    return _rh.magnitude
    

def cal_wind(u, v):
    """ # 根据风分量计算风速风向

        u：eastward component of the wind，向东的风速分量，m/s，单个值或n维数组
        v：northward component of the wind，向北的风速分量，m/s，单个值或n维数组

        return：ws，wd
            ws：风速，m/s
            wd：风向，风吹来的方向，0为北风，90为东风，180为南风

        例如：
            u = [1  0 -1  1  0 -1  1  0 -1]
            v = [1  0 -1  0 -1  1 -1  1  0]
        -> wd = [225. nan  45. 270. 0. 135. 315. 180.  90.]

        实际情况下，u和v不太可能同时为0，u、v同时为0的wd设为nan

        ---------------------------------------------------------
        MetPy库中有函数可以计算metpy.calc.wind_direction，
        u、v均为0时计算结果为0，风向为北风时计算结果为360

        from metpy.calc import wind_direction
        from metpy.units import units

        wd = wind_direction(u=u * units('m/s'), v=v*units('m/s'))

        -> wd = [225.   0.  45. 270. 360. 135. 315. 180.  90.]
        ---------------------------------------------------------

    2023-05-22 从ml_nc.py中转移，此后仅在本文件中更新
    单进程单线程
    """

    # 判断输入数据类型，并转换为np.ndarray处理
    if type(u) != type(v):
        raise TypeError('u、v数据类型不一致')

    elif isinstance(u, np.ndarray):
        uv = np.array((u, v))

    elif isinstance(u, pd.Series):
        uv = np.array((u.to_numpy(), v.to_numpy()))

    elif isinstance(u, list):
        uv = np.array((np.array(u), np.array(v)))

    elif isinstance(u, float) or isinstance(u, int):
        uv = np.array((u, v)).reshape(2, 1)

    else:
        raise TypeError('不支持的u、v值数据类型：%s' % type(u))

    # 提取u、v
    _u = uv[0, ...]
    _v = uv[1, ...]

    # 计算风速
    _ws = np.sqrt(_u ** 2 + _v ** 2)
    # print('ws:\n', ws)

    # 计算风向
    _wd = (np.arctan2(_u, _v) - np.arctan2(0, np.abs(_v))) * 180 / np.pi + 180
    _wd[_wd == 360] = 0

    # u、v同时为0的wd设为nan
    _index = np.all(uv == 0, axis=0)
    _wd[_index] = np.nan

    return _ws, _wd


def unit_conversion(array: np.ndarray, short_name: str):
    """ 从hdf5文件中提取数据时根据short_name转换单位 

        return: np.ndarray
    
    2023-09-18
    """
    
    # 类型判断
    if not isinstance(array, np.ndarray):
        raise TypeError('array must be a numpy.ndarray, your input: %s' % type(array))

    # 单位转换
    if short_name in ['t2m', 'd2m', 'skt', 'stl1']:
        array = array - 273.15    # K -> °

    elif short_name == 'tp':
        array = array * 1000  # m -> mm

    elif short_name == 'sp':
        array = array * 0.01  # Pa -> hPa

    else:
        pass

    # 返回数据
    return array


def cal_daily_era5(data: pd.DataFrame | pd.Series, std=False, sum=False):
    """ # 日均值/累计值计算,

        data: pd.DataFrame, 含有datetime时间索引, 秒/分钟/小时分辨率;
        std: 是否统计标准偏差
        sum: 是否计算日累计值

        return (mean, std) -> (pd.DataFrame | pd.Series, pd.DataFrame | pd.Series | None)

    单进程
    2023-09-17 v1
    """

    # 判断data类型
    if not isinstance(data, pd.DataFrame | pd.Series):
        raise ValueError("data类型错误: %s, 可选: pd.DataFrame | pd.Series" % type(data))

    """ Pandas中DatetimeIndex索引默认一天24h为00:00-23:00, 0:00的浓度应该是前一天23:00-0:00之间的均值, 也即前一天的值
        此处需要先将时间索引 - 1小时, 即00:00 - 23:00的值实际表示的是1:00 - 24:00的小时值 """

    # 时间补偿
    data.index -= pd.Timedelta(value=1, unit="H")

    # 按日分组
    group_1day = data.resample("D")

    """ 计算日均值/累计值 """
    if sum:
        # 累计值
        df_mean = group_1day.sum()
    else:
        # 日均值
        df_mean = group_1day.mean()

    # 删除空行
    df_mean.dropna(axis=0, how="all", inplace=True)

    """ 计算标准偏差 """    
    if std and not sum:
        df_std = group_1day.std()
        df_std.dropna(axis=0, how="all", inplace=True)
    else:
        df_std = None

    # 返回数据
    return df_mean, df_std


def cal_monthly_era5(data: pd.DataFrame | pd.Series, percentile=False, std=False, sum=False):
    """ # 月均值/累计值计算,
        
        data: pd.DataFrame, 含有datetime时间索引, 秒/分钟/小时/天分辨率;
        percentile: 统计均值时的百分位数, 0-100, 0表示最小值, 100表示最大值, 90表示90分位数, 默认为False(即求均值);
        std: 不统计百分位数时是否统计标准偏差
        sum: 统计累加值

        return (mean, std, percentile) -> (pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None)

    单进程
    2023-09-18 v1
    """

    """ 月统计 """
    # 按月分组
    group_1month = data.resample("MS")

    # 统计分位数
    if percentile:

        # 分位数
        df_percentile = group_1month.quantile(q=percentile / 100)

        # 返回数据
        return None, None, df_percentile

    # 统计月累加值
    elif sum is True:

        # 计算累计值
        df_sum = group_1month.sum()

        return df_sum, None, None

    # 统计月均值
    else:
        # 计算均值
        df_mean = group_1month.mean()

        # 计算标准偏差
        if std:
            df_std = group_1month.std()
        else:
            df_std = None
        
        # 返回数据
        return df_mean, df_std, None


def cal_seasonal_era5(data: pd.DataFrame | pd.Series, percentile=False, std=False, sum=False):
    """ # 季节均值/累计值计算,

        data: pd.DataFrame, 含有datetime时间索引,  秒/分钟/小时/天分辨率;
        percentile: 统计均值时的百分位数, 0-100, 0表示最小值, 100表示最大值, 90表示90分位数, 默认为False(即求均值);
        std: 不统计百分位数时是否统计标准偏差
        sum: 统计累加值

        return (mean, std, percentile) -> (pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None)

    单进程
    2023-09-18 v1
    """

    """ 季节统计 """
    # 按季节分组
    group_1season = data.resample("QS-MAR")

    # 统计分位数
    if percentile:

        # 分位数
        df_percentile = group_1season.quantile(q=percentile / 100)

        return None, None, df_percentile

    # 统计累加值
    elif sum is True:

        # 计算累计值
        df_sum = group_1season.sum()

        return df_sum, None, None
    
    # 统计均值
    else:
        # 计算均值
        df_mean = group_1season.mean()

        # 计算标准偏差
        if std:

            df_std = group_1season.std()

            return df_mean, df_std, None

        # 返回数据
        return df_mean, None, None


def cal_yearly_era5(data: pd.DataFrame | pd.Series, percentile=False, std=False, sum=False):
    """ # 年均值/累计值计算

        data: pd.DataFrame, 含有datetime时间索引, 秒/分钟/小时/天分辨率;
        percentile: 统计均值时的百分位数, 0-100, 0表示最小值, 100表示最大值, 90表示90分位数, 默认为False(即求均值);
        std: 不统计百分位数时是否统计标准偏差
        sum: 统计累加值

        return (mean, std, percentile) -> (pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None, 
                                            pd.DataFrame | pd.Series | None)
    单进程
    2023-09-18 v1
    """

    ''' 年统计 '''
    # 按年分组
    group_1year = data.resample('YS')

    # 统计百分位数
    if percentile:

        # 分位数
        df_percentile = group_1year.quantile(q=percentile / 100)

        # 返回数据
        return None, None, df_percentile
    
    # 统计累加值
    elif sum is True:

        # 计算累计值
        df_sum = group_1year.sum()

        return df_sum, None, None
    
    else:
        # 计算均值
        df_mean = group_1year.mean()
        
        # 计算标准偏差
        if std:
            df_std = group_1year.std()

            return df_mean, df_std, None

        # 返回数据
        return df_mean, None, None


def cal_era5_cities_by_species(data_: pd.DataFrame, time_resolution: str, dict_city: dict, percentile=None, sum=False):
    """ 整理ERA5数据 -> 城市级别统计数据

        data_: 各站点小时均值数据
        time_resolution: 数据输出时间分辨率(重要), 'YS'代表年均值, 'QS-MAR' 代表季节(3-5为春天), 'MS'代表月, 'D'代表天, 'H'代表小时;
        dict_city: {'北京市': ['1001A', '1003A', ...], '保定市': ['1051A', '1052A', ...], ...}
        percentile: 统计城市浓度时, 是否使用评价浓度, percentile=90表示取90分位数;
        sum: 是否计算累计值

        return: pd.DataFrame
    
    单进程
    2023-09-18 v1
    """

    # 城市列表
    list_city_all = list(dict_city.keys())

    if time_resolution == 'H':

        list_series_city = []
        for city in list_city_all:
            # 站点列表
            list_sites_of_city = [site for site in dict_city[city] if site in data_.columns]
            # print('list_sites_of_city:', list_sites_of_city)

            if list_sites_of_city:
                # 待求平均的数据
                data_to_mean = data_.loc[:, list_sites_of_city]

                # 去除空行
                data_to_mean.dropna(how='all', axis=0, inplace=True)

                # 求平均
                series_city = pd.Series(data=np.nanmean(data_to_mean.to_numpy(), axis=1),
                                        index=data_to_mean.index,
                                        name=city)

                list_series_city.append(series_city)

        # 合并各城市数据
        df_hourly = pd.concat(objs=list_series_city, axis=1)

        # 返回数据
        return df_hourly

    # 计算日均值/累计值
    df_data = cal_daily_era5(data=data_, std=False, sum=sum)[0]

    # 非空值城市列表
    list_city_export = []

    # 将城市包含的站点 -> 城市对应的DataFrame {'北京市': df1, '保定市': df2, ...}，并统计为城市日均值
    list_df_city = []
    for city in list_city_all:
        # 包含的站点列表
        list_sites_of_city = [site for site in dict_city[city] if site in df_data.columns]
        # print('list_sites_of_city:', list_sites_of_city)

        if list_sites_of_city:
            # 待统计的数据
            data_to_mean = df_data.loc[:, list_sites_of_city]

            # 去除空行
            data_to_mean.dropna(how='all', axis=0, inplace=True)

            # 计算不同站点的平均值-得到城市数据
            series_city = pd.Series(data=np.nanmean(data_to_mean.to_numpy(), axis=1),
                                    index=data_to_mean.index,
                                    name=city,
                                    )

            # 城市数据添加至列表
            list_df_city.append(series_city)

            # 城市名称添加至列表
            list_city_export.append(city)

    # 合并各城市数据
    df_data = pd.concat(objs=list_df_city, axis=1)

    # 表头 -> 城市名
    df_data.columns = list_city_export

    # 再计算对应分辨率的均值
    if time_resolution == "MS":

        if percentile:
            df_data = cal_monthly_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_monthly_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    elif time_resolution == "QS-MAR":

        if percentile:
            df_data = cal_seasonal_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_seasonal_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    elif time_resolution == "YS":

        if percentile:
            df_data = cal_yearly_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_yearly_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    else:
        pass

    # 返回数据
    return df_data


def cal_era5_sites_by_species(data_: pd.DataFrame, time_resolution: str, percentile=None, sum=False):
    """ 整理ERA5数据 -> 站点级别统计数据

        data_: 各站点小时均值数据
        time_resolution: 数据输出时间分辨率(重要), 'YS'代表年均值, 'QS-MAR' 代表季节(3-5为春天), 'MS'代表月, 'D'代表天, 'H'代表小时;
        dict_city: {'北京市': ['1001A', '1003A', ...], '保定市': ['1051A', '1052A', ...], ...}
        percentile: 统计城市浓度时, 是否使用评价浓度, percentile=90表示取90分位数;
        sum: 是否计算累计值

        return: pd.DataFrame
    
    单进程
    2023-09-18 v1
    """

    if time_resolution == 'H':

        # 返回数据
        return data_

    # 计算日均值/累计值
    df_data = cal_daily_era5(data=data_, std=False, sum=sum)[0]

    # 再计算对应分辨率的均值
    if time_resolution == "MS":

        if percentile:
            df_data = cal_monthly_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_monthly_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    elif time_resolution == "QS-MAR":

        if percentile:
            df_data = cal_seasonal_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_seasonal_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    elif time_resolution == "YS":

        if percentile:
            df_data = cal_yearly_era5(
                data=df_data,
                percentile=percentile,
                std=False,
                sum=False,
            )[-1]
        else:
            df_data = cal_yearly_era5(
                data=df_data,
                percentile=False,
                std=False,
                sum=sum,
            )[0]

    else:
        pass

    # 返回数据
    return df_data


class ReadEra5(QThread):
    """ 从gkd数据库(*.h5)读取数据，并统计

        dir_h5_land: hdf5文件所在目录-ERA5 Land
        dir_h5_single_levels: hdf5文件所在目录-ERA5 Single Levels
        species_land: ERA5 Land变量名列表, t2m、d2m、tp、sp、u10、v10 ...
        species_single_levels: ERA5 Single Levels变量名列表, u100、v100、blh ...
        date_range: 日期范围, ['2014-05-13', '2020-10-28']
        dict_percentile: dict, {'O3': 90, ...}, 计算年均浓度时, 是否使用评价浓度, percentile=90 表示取90分位数;
        site_info: 站点信息类 SiteInfo()
        list2export: 待导出省/地市/站点列表
        site_type: 在统计省/地市数据时考虑的站点类型, 'urban'、'background'、'both'
        time_resolution: 数据输出时间分辨率, 'YS':年均值, 'QS-MAR':季节(3-5为春天), 'MS':月, 'D':天, 'H':小时, 'S':秒;
        region_level: 统计区域类型, 'province'、'city'、'site'

    线程池ThreadPoolExecutor
    2023-09-18 v1
    """

    # 信号槽：状态值 {'text': , 'value': }
    signal_progress = pyqtSignal(dict)

    # 信号槽：错误信息
    signal_error = pyqtSignal(str)

    def __init__(self,
                 dir_h5_land: str,
                 dir_h5_single_levels: str,
                 species_land: list,
                 speices_single_levels: list,
                 date_range: list,
                 dict_percentile: dict,
                 site_info,
                 list2export: list,
                 site_type='urban',
                 time_resolution='H',
                 region_level='city',
                 ):

        super().__init__()

        self.dir_h5_land = dir_h5_land
        self.dir_h5_single_levels = dir_h5_single_levels
        self.species_land = species_land
        self.species_single_levels = speices_single_levels
        self.date_range = date_range
        self.dict_percentile = dict_percentile
        self.si = site_info
        self.list2export = list2export
        self.site_type = site_type
        self.time_resolution = time_resolution
        self.region_level = region_level

        # 站点归属
        self.site_dict()
        # print(self.dict_sites)

        # 最终数据
        self.data = None

    def site_dict(self):
        """ 整理站点数据，归类到省市 """

        if self.region_level == 'province':
            self.dict_sites, self.sites = gkd.get_site_dict(provinces=self.list2export, site_type=self.site_type, site_info=self.si)
        elif self.region_level == 'city':
            self.dict_sites, self.sites = gkd.get_site_dict(cities=self.list2export, site_type=self.site_type, site_info=self.si)
        elif self.region_level == 'site':
            self.dict_sites, self.sites = gkd.get_site_dict(sites=self.list2export, site_info=self.si)

    def cal_sites(self):
        """ 统计站点数据 """

        # # 线程池初始化
        # pool = ThreadPoolExecutor(max_workers=8)

        # # 准备参数
        # args_ = []
        # for s in self.species_land:

        #     # 百分位数
        #     percentile_s = self.dict_percentile[s] if s in self.dict_percentile.keys() else None

        #     # 计算累计值的变量: 降雨量tp
        #     if s == 'tp':
        #         sum_ = True
        #     else:
        #         sum_ = False

        #     args_.append((self.data[s], self.time_resolution, self.dict_sites, percentile_s, sum_))

        # # 提交任务
        # list_result = pool.map(cal_era5_cities_by_species, *zip(*args_))

        # # 提取结果
        # self.data = dict(zip(self.species_land, list_result))

    def cal_cities(self):
        """ 统计城市数据 """

        # 进程池初始化
        pool = ThreadPoolExecutor(max_workers=8)

        # 物种列表
        list_species= list(self.data.keys())

        # 准备参数
        args_ = []
        for s in list_species:

            # 百分位数
            percentile_s = self.dict_percentile[s] if s in self.dict_percentile.keys() else None

            # 计算累计值的变量: 降雨量tp
            if s == 'tp':
                sum_ = True
            else:
                sum_ = False

            args_.append((self.data[s], self.time_resolution, self.dict_sites, percentile_s, sum_))

        # 提交任务
        list_result = list(pool.map(cal_era5_cities_by_species, *zip(*args_)))

        # 结果
        self.data = dict(zip(list_species, list_result))

    def cal_regions(self):
        """ 统计区域数据 """

        # # 进程池初始化
        # pool = ThreadPoolExecutor(max_workers=8)

        # # 准备参数
        # args_ = []
        # for s in self.species_land:

        #     # 百分位数
        #     percentile_s = self.dict_percentile[s] if s in self.dict_percentile.keys() else None

        #     if s == 'O3':
        #         args_.append((self.data[s], self.time_resolution, self.o3_mda8, self.dict_stat_standard, self.dict_sites, percentile_s))
        #     else:
        #         args_.append((self.data[s], self.time_resolution, False, self.dict_stat_standard, self.dict_sites, percentile_s))

        # # 提交任务
        # list_result = list(pool.map(cal_gkd_regions_by_species, *zip(*args_)))

        # # 结果
        # self.data = dict(zip(self.species_land, list_result))

    def run(self):

        try:

            """ 读取数据 """
            # 发送状态信号
            self.signal_progress.emit({'text': '读取数据...', 'value': None})

            # 调用函数读取数据ERA5 Land
            data_land = hdf5.read_hdf5_by_sites(dir_h5=self.dir_h5_land, species=self.species_land, date_range=self.date_range, sites=self.sites)
            # print(data_land, 'dddd')
            
            # 调用函数读取数据ERA5 Land 
            data_single_levels = hdf5.read_hdf5_by_sites(dir_h5=self.dir_h5_single_levels, species=self.species_single_levels, date_range=self.date_range, sites=self.sites)
            # print(data_single_levels)

            # 合并数据
            self.data = {**data_land, **data_single_levels}

            """ 整理数据 """
            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计...', 'value': None})

            if self.region_level == 'site':
                self.cal_sites()
            elif self.region_level == 'city':
                self.cal_cities()
            elif self.region_level == 'province':
                self.cal_regions()
            else:
                raise ValueError('暂不支持的导出区域级别！')

            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计完成！', 'value': None})

            # 延迟0.5s
            self.msleep(500)

            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计完成！', 'value': 1000, 'source': 'ReadGkd'})

        except:
            # 错误信息返回
            self.signal_error.emit(traceback.format_exc())


class ReadEra5Site(QThread):
    """ 从gkd数据库(*.h5)读取站点数据，并统计

        dir_h5_land: hdf5文件所在目录-ERA5 Land
        dir_h5_single_levels: hdf5文件所在目录-ERA5 Single Levels
        species_land: ERA5 Land变量名列表, t2m、d2m、tp、sp、u10、v10 ...
        species_single_levels: ERA5 Single Levels变量名列表, u100、v100、blh ...
        date_range: 日期范围, ['2014-05-13', '2020-10-28']
        dict_percentile: dict, {'O3': 90, ...}, 计算年均浓度时, 是否使用评价浓度, percentile=90 表示取90分位数;
        site_info: 站点信息类 SiteInfo()
        list2export: 待导出省/地市/站点列表
        time_resolution: 数据输出时间分辨率, 'YS':年均值, 'QS-MAR':季节(3-5为春天), 'MS':月, 'D':天, 'H':小时, 'S':秒;
        region_level: 统计区域类型, 'province'、'city'、'site'

    线程池ThreadPoolExecutor
    2024-07-09 v1
    """

    # 信号槽：状态值 {'text': , 'value': }
    signal_progress = pyqtSignal(dict)

    # 信号槽：错误信息
    signal_error = pyqtSignal(str)

    def __init__(self,
                 dir_h5_land: str,
                 dir_h5_single_levels: str,
                 species_land: list,
                 speices_single_levels: list,
                 date_range: list,
                 dict_percentile: dict,
                 site_info,
                 list_sites: list,
                 time_resolution='H',
                 ):

        super().__init__()

        self.dir_h5_land = dir_h5_land
        self.dir_h5_single_levels = dir_h5_single_levels
        self.species_land = species_land
        self.species_single_levels = speices_single_levels
        self.date_range = date_range
        self.dict_percentile = dict_percentile
        self.si = site_info
        self.sites = list_sites
        self.time_resolution = time_resolution

        # 最终数据
        self.data = None

    def cal_sites(self):
        """ 统计站点数据 """

        # 进程池初始化
        pool = ThreadPoolExecutor(max_workers=8)

        # 物种列表
        list_species= list(self.data.keys())

        # 准备参数
        args_ = []
        for s in list_species:

            # 百分位数
            percentile_s = self.dict_percentile[s] if s in self.dict_percentile.keys() else None

            # 计算累计值的变量: 降雨量tp
            if s == 'tp':
                sum_ = True
            else:
                sum_ = False

            args_.append((self.data[s], self.time_resolution, percentile_s, sum_))

        # 提交任务
        list_result = list(pool.map(cal_era5_sites_by_species, *zip(*args_)))

        # 结果
        self.data = dict(zip(list_species, list_result))

    def run(self):

        try:

            """ 读取数据 """
            # 发送状态信号
            self.signal_progress.emit({'text': '读取数据...', 'value': None})

            # 调用函数读取数据ERA5 Land
            data_land = hdf5.read_hdf5_by_sites(dir_h5=self.dir_h5_land, species=self.species_land, date_range=self.date_range, sites=self.sites)
            # print(data_land, 'dddd')
            
            # 调用函数读取数据ERA5 Land 
            data_single_levels = hdf5.read_hdf5_by_sites(dir_h5=self.dir_h5_single_levels, species=self.species_single_levels, date_range=self.date_range, sites=self.sites)
            # print(data_single_levels)

            # 合并数据
            self.data = {**data_land, **data_single_levels}

            """ 整理数据 """
            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计...', 'value': None})

            # 调用函数整理数据
            self.cal_sites()

            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计完成！', 'value': None})

            # 延迟0.5s
            self.msleep(500)

            # 发送状态信号
            self.signal_progress.emit({'text': '数据统计完成！', 'value': 1000, 'source': 'ReadGkd'})

        except:
            # 错误信息返回
            self.signal_error.emit(traceback.format_exc())


def clip_cal(data_grid: xr.Dataset, data_shp: gpd.geodataframe):
    """ # 用shp数据对网格数据进行裁剪并计算均值

        data_grid: xr.Dataset，含有参考系的网格数据，包含x、y、time、spatial_ref维度
        data_shp: gpd.geodataframe，含有参考系的shp数据

        return: pd.Series，沿着时间轴，将shp代表的区域内的数据进行平均

    2023-05-24 v1
    单进程单线程
    """

    # 变量名
    short_name = [i for i in data_grid.data_vars][0]

    # 裁剪
    clip_r = data_grid.rio.clip(data_shp.geometry.apply(mapping), data_shp.crs, drop=False)

    # 计算均值
    ds_r_mean = clip_r[short_name].mean(dim=['x', 'y'])
    series_r = pd.Series(data=ds_r_mean.to_numpy(), index=ds_r_mean.coords['time'], name=short_name)

    return series_r


def stat_era5_without_levels_by_region(dir_data: os.PathLike, data_shp: gpd.GeoDataFrame, utc_offset=8, cpu=4, freq_rule='QS'):
    """ # 分区域统计ERA5 land或ERA5 single levels数据（无level维度）

        dir_data: 存放ERA5 land或ERA5 single levels数据某个变量的目录
            用xarray库读取为xr.Dataset对象

        data_shp：用于裁剪的shp数据
            多边形裁剪区域，用geopandas库读取为GeoDataframe对象，必须包含name列
            将用shp中的指定区域（用name列进行区分）进行裁剪，然后进行区域平均

        utc_offset：时区，默认为=8，即北京时间

        cpu: 调用几个核心进行处理

        freq_rule: 计算周期的长度，AS-每年、QS-每三个月、MS-每月、D-每天

        return: dict, 如：{
                           'short_name': 't2m',
                           'data': pd.DataFrame(),  # 时间序列，列名为data_shp中的name名
                          }

    """

    from concurrent.futures import ProcessPoolExecutor

    if 'name' not in data_shp.columns:
        raise ValueError('data_shp中不含name列')

    # 赋值
    _gdf = data_shp

    # 省份/城市列表
    list_region = _gdf.loc[:, 'name'].to_list()

    # ERA5文件列表
    list_files = [os.path.join(dir_data, i) for i in os.listdir(dir_data) if i.endswith('.nc')]

    # 读取ERA5数据（决速步骤1）
    ds_ori = xr.open_mfdataset(paths=list_files, engine='rasterio')

    # 时间转换
    ds_ori['time'] = ds_ori['time'].to_index().to_datetimeindex()

    # 变量名
    short_name = [i for i in ds_ori.data_vars][0]

    # 写入坐标系
    ds_ori.rio.write_crs('epsg:4326', inplace=True)

    # 按区域裁剪及计算均值
    dict_region2clip = {key: [] for key in list_region}

    # 将df_ori按日期分割
    for t_, ds_ in ds_ori.resample(time=freq_rule):

        # time转换为字符串时间
        dt_string = ds_['time'].to_index().strftime('%Y.%m.%d')

        print('变量：%s，计算周期：%s-%s' % (short_name, dt_string[0], dt_string[-1]), end='，')

        t_start = time.time()

        # dataset载入内存
        ds_.compute()

        pool = ProcessPoolExecutor(max_workers=cpu)
        list_result_ds = []
        for r in list_region:
            # 获取区域r
            gdf_r = _gdf[_gdf['name'] == r]

            # series_r = clip_cal(gdf_r, ds_)
            series_r = pool.submit(clip_cal, ds_, gdf_r)

            list_result_ds.append(series_r)

        pool.shutdown(wait=True)

        # 提取数据
        list_result_ds = [i.result() for i in list_result_ds]
        dict_result_ds = dict(zip(list_region, list_result_ds))

        for r in list_region:
            dict_region2clip[r].append(dict_result_ds[r])

        t_end = time.time()

        print('完成！耗时：%.1f min' % ((t_end - t_start) / 60))

    print('合并数据...', end=' ')

    # 合并数据
    list_series_r = []
    for r in dict_region2clip.keys():
        series_r = pd.concat(objs=dict_region2clip[r], axis=0, join='outer')
        series_r.name = r
        list_series_r.append(series_r)

    # 不同区域pd.Series合并为pd.DataFrame
    _df = pd.concat(objs=list_series_r, axis=1, join='outer')
    _df.columns = list_region
    _df.index.name = 'datetime'

    # print(_df)
    # print(_df.index)

    """ 还原ERA5中的累计数据（如tp、str、strd、ssr、ssrd...）为小时值，
        以降雨量为例，转换降雨量数据累计值为小时值

        ERA5_Land_Hourly中的小时降雨量数据实际上是累积值，
            从第1天1:00开始，第2天0:00结束，1:00代表的是实际降水量，2:00代表的是1:00至2:00的累计降水量，
            以此类推，第2天0:00代表的是第1天的日降水量 """

    if short_name in ['tp', 'str', 'strd', 'ssr', 'ssrd']:
        # 提取1:00的数据
        df_hour1 = _df[_df.index.hour == 1]

        # 按照时间轴的方向，后面的数据减去前面的数据，第1个时刻的数据摒弃
        _df = _df.diff(periods=1, axis=0)

        # 因为1:00的数据不用减去前面时刻的值，因此将前面提取出来的1:00时刻的值还原
        _df[_df.index.hour == 1] = df_hour1

    """ 删除nan行 """
    _df.dropna(how='all', axis=0, inplace=True)

    """ 时间校正, UTC -> UTC+8 (北京时间) """
    _df.index = _df.index + pd.Timedelta(hours=utc_offset)

    """ 单位转换
        t2m, d2m, skt, stl1: K --> °C;
        tp: total precipitation, m --> mm
        sp: surface pressure, Pa --> hPa """

    # 温度
    if short_name in ['t2m', 'd2m', 'skt', 'stl1']:
        _df = _df - 273.15

    # 降水量，m -> mm
    elif short_name == 'tp':
        _df = _df * 1000

    # 压力，Pa -> hPa
    elif short_name == 'sp':
        _df = _df * 0.01

    else:
        pass

    print('完成！\n\n', _df)

    # 返回数据
    return {'data': _df, 'short_name': short_name}


def stat_era5_with_levels_by_region(dir_data: os.PathLike, data_shp: gpd.GeoDataFrame, utc_offset=8, levels=[850, 1000]):
    """ # 分区域统计ERA5 pressure levels数据（含有level维度）

        dir_data: ERA5 pressure levels数据某变量的存放路径

        data_shp：用于裁剪的shp数据
            多边形裁剪区域，用geopandas库读取为GeoDataframe对象，必须包含name列
            将用shp中的指定区域（用name列进行区分）进行裁剪，然后进行区域平均

        utc_offset：时区，默认为=8，即北京时间

        level: list，需要读取的pressure levels

        return: dict, 如：{
                           'short_name': 't',
                           'data': [pd.DataFrame(), ...],  # 时间序列，列名为site code
                          }

        """

    if 'name' not in data_shp.columns:
        raise ValueError('data_shp中不含name列')

    # 赋值
    _gdf = data_shp

    # 省份/城市列表
    list_region = _gdf.loc[:, 'name'].to_list()

    # ERA5文件列表
    list_files = [os.path.join(dir_data, i) for i in os.listdir(dir_data) if i.endswith('.nc')]

    # 读取ERA5数据（决速步骤1）
    # ds_ori = xr.open_mfdataset(paths=list_files, engine='rasterio')
    ds_ori = xr.open_mfdataset(paths=list_files, engine='netcdf4')

    # 经纬度重命名
    ds_ori = ds_ori.rename({'latitude': 'y', 'longitude': 'x'})

    # 变量名
    short_name = [i for i in ds_ori.data_vars][0]

    # 写入坐标系
    ds_ori.rio.write_crs('epsg:4326', inplace=True)

    # 区域及level筛选
    list_df = []
    for level in levels:
        ds_level = ds_ori.sel(level=level)

        # 按区域裁剪
        dict_region2clip = dict()
        for r in list_region:
            # 获取区域r
            gdf_r = _gdf[_gdf['name'] == r]

            # 裁剪
            clip_r = ds_level.rio.clip(gdf_r.geometry.apply(mapping), gdf_r.crs, drop=False)

            dict_region2clip[r] = clip_r

        # 分区域计算均值
        list_series_r = []
        for r in list_region:
            clip_r = dict_region2clip[r]

            # 平均
            ds_r_mean = clip_r[short_name].mean(dim=['x', 'y'])

            # 转换为pd.Series
            series_r = pd.Series(data=ds_r_mean.to_numpy(), index=ds_r_mean.coords['time'], name=short_name)

            list_series_r.append(series_r)

        # 不同区域pd.Series合并为pd.DataFrame
        df_level = pd.concat(objs=list_series_r, axis=1, join='outer')
        df_level.columns = list_region
        df_level.index.name = 'datetime'

        """ 删除nan行 """
        df_level.dropna(how='all', axis=0, inplace=True)

        """ 时间校正, UTC -> UTC+8 (北京时间) """
        df_level.index = df_level.index + pd.Timedelta(hours=utc_offset)

        """ 单位转换
            t: K --> °C;
        """

        # 温度
        if short_name in ['t']:
            df_level = df_level - 273.15

        else:
            pass

        print('\n\n', df_level)
        list_df.append(df_level)

    # 返回数据
    return {'data': list_df, 'short_name': short_name}


if __name__ == '__main__':
    dir_land = "E:\\bigdata\\era5_netcdf\\site,era5_land"
    dir_single = "E:\\bigdata\\era5_netcdf\\site,era5_single_levels"
    dir_pressure = ""
    # dir_pressure = "E:\\bigdata\\era5_netcdf\\site,era5_pressure_levels"

    list_land = ['d2m', 'skt', 'sp', 'ssrd', 'strd', 't2m', 'tp', 'u10', 'v10']
    list_single = ['blh', 'u100', 'v100']
    list_pressure = []

    t0 = time.time()
    df = mean_sites2city(
        dir_land=dir_land,
        dir_single=dir_single,
        dir_pressure=dir_pressure,
        # list_land=list_land,
        # list_single=list_single,
        # list_pressure=list_pressure,
        list_site=['1001A', '1003A'],
        time_resolution='D',
    )

    t1 = time.time()
    print(t1 - t0)
