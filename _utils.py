from __future__ import annotations
from typing import List, Tuple, Dict, Union, Sequence, Iterable
import os
import time
# import json
# import requests
import numpy as np
import pandas as pd
import h5py
# import pyproj
from pyproj import Geod
from pathlib import Path
from numpy.lib.stride_tricks import as_strided
# import shapely
# from shapely.geometry import MultiLineString
from shapely import get_coordinates, within, points, polygons
from shapely.geometry import LineString, MultiLineString, Point, Polygon
# from geopy.distance import great_circle, geodesic
from sklearn.neighbors import BallTree
# from ml_io import read_config
import geopandas as gpd
# from shapely.geometry import Point, Polygon
# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# from scipy.spatial import KDTree
# import pygeos


def geodistance(latlon1: Tuple[float, float], latlon2: Tuple[float, float]):
    """ 计算地球表面两个点之间的精确距离

        latlon1: 第1个点的经纬度，格式：(lat, lon)，单位：°
        latlon2: 第2个点的经纬度，格式：(lat, lon)，单位：°

        return: 距离，单位km

    方法1：geopy.great_circle()计算球面距离时假设地球为完美球体，不够准确；
        dist = great_circle(latlon1, latlon2).km

    方法2：geopy.geodesic()计算球面距离时假设地球为椭球体，最精确；
        dist = geodesic(latlon1, latlon2).km

    方法3：pyproj.Geod().inv()计算球面距离时假设地球为椭球体，准确性同geopy.geodesic，但速度快2个数量级；
        geod = pyproj.Geod(ellps='WGS84')
        dist = geod.inv(lons1=latlon1[1], lats1=latlon1[0], lons2=latlon2[1], lats2=latlon2[0])[2] * 0.001

    2022.10.09  v1 使用geopy.great_circle()
    2024.08.22  v2 使用pyproj.Geod().inv()
    单进程单线程
    """

    # Geod方法初始化
    geod = Geod(ellps='WGS84')
    
    # 计算距离
    dist = geod.inv(lons1=latlon1[1], lats1=latlon1[0], lons2=latlon2[1], lats2=latlon2[0])[2] * 0.001
    
    # 返回数据
    return dist


def geodistances(latlon1: Tuple[float, float], latlons: np.ndarray):
    """ 计算地球表面1个点与其它多个点之间的精确距离
        
        latlon1: 1个点的经纬度，格式：(lat, lon)，单位：°
        latlons: 其它多点的经纬度，2维数组(n*2)，第1、2列分别为lat和lon，单位：°

        return: 距离组成的1维数组，单位：km

    2022.10.31  v1.0  使用geopy.great_circle()
    2024.08.22  v2.0  使用pyproj.Geod().inv()
    单进程单线程
    """

    # Geod方法初始化
    geod = Geod(ellps='WGS84')

    # 准备数据
    lats1 = np.full(shape=(latlons.shape[0],), fill_value=latlon1[0])
    lons1 = np.full(shape=(latlons.shape[0],), fill_value=latlon1[1])
    
    # 计算距离
    dist = geod.inv(lons1=lons1, lats1=lats1, lons2=latlons[:, 1], lats2=latlons[:, 0])[2] * 0.001
    
    # 返回数据
    return dist


class NearbyCoords:

    """ 查询1个或多个坐标最近的国控点
    
        dict_code2coordinate: 用于查询的字典，key=站点代码，value=(latitude, longitude)
    
    2024-08-22 v1
    """

    def __init__(self, dict_code2coordinate=None):

        # 站点代码
        self.arr1d_code = np.array(list(dict_code2coordinate.keys()))

        # 站点坐标，第1列为经度，第2列为纬度
        self.arr2d_coords = np.array([dict_code2coordinate[i] for i in self.arr1d_code])[:, ::-1]

        # 站点坐标转换为弧度
        self.arr2d_coords_rad = np.radians(self.arr2d_coords)

        # 构建BallTree对象，使用haversine作为距离度量，用于快速粗略搜索
        self.tree = BallTree(self.arr2d_coords_rad, metric='haversine')

        # 初始化pyproj.Geod对象，用于计算精确的球面距离
        self.geod = Geod(ellps='WGS84')

        # 地球半径
        # self.radius_earth = 6371.0

    def querycoord(self, lonlat: Tuple[float, float], radius: float):
        """ 查询1个坐标，用于替代get_nearest_gkd_coordinates函数
            lonlat: (lon, lat)，°
            radius: 搜索半径，单位km
        
        2024-08-22 v1
        """

        # 1.5倍半径快速搜索，获得索引
        ind1 = self.tree.query_radius(X=np.radians([lonlat]), r=radius * 1.5 / 6371.0, return_distance=False)[0]

        if ind1.size == 0:
            return {}

        # 待精确计算的坐标数组
        arr2d_coords = self.arr2d_coords[ind1]

        # 计算精确球面距离
        arr1d_dist = np.array([self.geod.inv(lonlat[0], lonlat[1], i[0], i[1])[2] * 0.001 for i in arr2d_coords])

        # 返回符合条件的索引
        ind2 = np.where(arr1d_dist <= radius)[0]
        
        if ind2.size == 0:
            return {}

        # 准备返回数据
        dict_result = {}

        for i in ind2:
            dict_result[self.arr1d_code[ind1][i]] = {
                'coord': arr2d_coords[i][::-1],
                'dist': arr1d_dist[i],   
            }

        # 返回数据
        return dict_result

    def querycoords(self, lonlats: np.ndarray, radius: float):
        """ 查询多个坐标
            lonlats: np.ndarray，维度：n*2，第1、2列分别为经度、纬度，°
            radius: 搜索半径，单位km
        
        2024-08-22 v1
        """
        pass


def create_grid_polygons(array2d_lat, array2d_lon):
# def create2dPolygons(array2d_lat, array2d_lon):
    """ 将输入的相邻网格顶点连接起来, 生成对应网格的多边形数组
        即使用griddesc2mesh函数返回的数据生成清单中每个网格代表的多边形区域数组

        array2d_lat: 纬度组成的2维数组 (m+1)*(n+1)
        array2d_lon: 经度组成的2维数组 (m+1)*(n+1)

        return: 2维数组 m*n

    单进程单线程
    2024-06-24 v1
    """

    # 输入数组维度
    nrows, ncols = array2d_lat.shape

    """ 循环读取网格顶点坐标，连接后生成多边形 """
    # 利用 as_strided 创建滑动窗口视图
    strided_view_lon = as_strided(
        x=array2d_lon, 
        shape=(nrows - 1, ncols - 1, 2, 2),
        strides=(*array2d_lon.strides, *array2d_lon.strides),          
        writeable=False)
    
    strided_view_lat = as_strided(
        x=array2d_lat, 
        shape=(nrows - 1, ncols - 1, 2, 2),
        strides=(*array2d_lat.strides, *array2d_lat.strides),
        writeable=False)
    
    # 顶点经度、维度
    new_array_lon = strided_view_lon.reshape(-1, 4)
    new_array_lat = strided_view_lat.reshape(-1, 4)

    # 第3、4列互换位置
    new_array_lon[:, [2, 3]] = new_array_lon[:, [3, 2]]
    new_array_lat[:, [2, 3]] = new_array_lat[:, [3, 2]]

    # 合并经纬度为坐标
    list_vertex = np.dstack((new_array_lon, new_array_lat))

    # 生成多边形数组
    array2d_polygons = polygons(geometries=list_vertex).reshape(nrows - 1, ncols - 1)

    # 返回数据
    return array2d_polygons


# def countPointInPolygon(line: LineString, polygon: Polygon):
#     """ 统计LineString中有多少个点位于Polygon内部 

#         line: shapely LineString，一条线
#         polygon: shapely Polygon，一个多变形

#     2024-08-31 v1
#     """

#     return sum(polygon.contains(point) for point in [Point(coord) for coord in line.coords])


# def countPointInPolygon(line: pygeos.geometry, polygon: pygeos.geometry):
#     """ 统计linestring中有多少个点位于polygon内部 

#         line_geos: pygeos LINESTRING，一条线
#         polygon: pygeos POLYGON，一个多变形

#     2024-09-02 v2 用pygeos库替换shapely库
#     """

#     # 提取坐标
#     coords = pygeos.get_coordinates(line)

#     # 转换为points
#     points = pygeos.points(coords)

#     # 计算有多少点位于多边形内
#     points_in_polygon = pygeos.within(points, polygon).sum()

#     # 返回数据
#     return points_in_polygon


def countPointInPolygon(line: LineString, polygon: Polygon):
    """ 统计linestring中有多少个点位于polygon内部 

        line: shapely.geometry.LineString，一条线
        polygon: shapely.geometry.Polygon，一个多变形

    2024-09-04 v3   用shapely库替换pygeos库
    """

    # 提取坐标
    arr2d_coords = get_coordinates(line)

    # 转换为points
    arr1d_points = points(arr2d_coords)

    # 计算有多少点位于多边形内
    points_in_polygon = within(arr1d_points, polygon).sum()

    # 返回数据
    return points_in_polygon


def weight(x):
    """ 权重函数
        x <= 10:        w=0.05
        10 < x <= 20:   w=0.42
        20 < x <= 80:   w=0.70
        x > 80:         w=1.0

    2024-08-30 v1
    """

    y = np.where(
        x <= 10, 0.05,
        np.where(x <= 20, 0.42,
                 np.where(x <= 80, 0.70, 1.0)
                 )
                 )

    return y


def wpscf(path_shp: str | Path, name: str, threshold: float, resolution: float = 1.0, extent: tuple[float, float, float, float] = None):
    """ 计算PSCF和WPSCF值 

        path_shp: 包含后向轨迹线的shp文件路径
        name: 浓度列名
        threshold: 阈值
        resolution: 网格分辨率，°
        extent: 网格范围，(lon_west, lat_south, lon_east, lat_north)

    2024-08-30 v1
    2024-09-04 v2   添加extent参数
    """

    # 读取包含后向轨迹线的shp文件
    gdf_lines = gpd.read_file(path_shp, engine='pyogrio')

    if extent is None:
        # 合并所有LineString到单一的MultiLineString
        combined = MultiLineString(gdf_lines.geometry.tolist())

        # 计算边界坐标
        lon_west, lat_south, lon_east, lat_north = combined.bounds
    
    else:
        lon_west, lat_south, lon_east, lat_north = extent

    # 生成经纬度数组
    arr1d_lon = np.arange(lon_west, lon_east + resolution, resolution)
    arr1d_lat = np.arange(lat_north, lat_south - resolution, -resolution)

    # 生成2维网格经纬度
    array2d_lon_m, array2d_lat_m = np.meshgrid(arr1d_lon, arr1d_lat)

    # 生成2维网格多边形
    array2d_polygons = create_grid_polygons(array2d_lat_m, array2d_lon_m)

    # 建立字典，用于方法计算结果
    list_var = ['Nij', 'Mij', 'PSCF', 'WPSCF']

    # 初始化字典
    dict_result = {}

    # 将ID存入字典
    dict_result['ID'] = np.arange(1, array2d_polygons.size + 1)

    for i in list_var:
        dict_result[i] = []

    # 遍历多边形
    for polygon in array2d_polygons.flatten():
        
        # 判断多边形与combined是否相交
        # if not polygon.intersects(combined):
        #     for i in list_var:
        #         dict_result[i].append(np.nan)
        #     continue

        # 与多边形相交的线型数据
        gdf_intersect = gdf_lines[gdf_lines.intersects(polygon)]

        # 计算有多少条轨迹经过该多边形，Nij
        Nij = gdf_intersect.shape[0]
        if Nij == 0:
            for i in list_var:
                dict_result[i].append(np.nan)
            continue

        # 计算其中有多少条轨迹对应的浓度值大于阈值，Mij
        Mij = gdf_intersect[gdf_intersect[name] > threshold].shape[0]
        if Mij == 0:
            for i in list_var:
                dict_result[i].append(np.nan)
            continue

        # 计算PSCF值
        PSCFij = Mij / Nij

        # 计算WPSCF值
        WPSCFij = PSCFij * weight(Nij)

        # 将PSCF值和WPSCF值存入列表
        dict_result['Nij'].append(Nij)
        dict_result['Mij'].append(Mij)
        dict_result['PSCF'].append(PSCFij)
        dict_result['WPSCF'].append(WPSCFij)
    
    # 将数据存入GeoDataFrame
    gdf_pscf = gpd.GeoDataFrame(
        data=dict_result,
        geometry=array2d_polygons.flatten(),
        )

    # 返回数据
    return gdf_pscf


# def wcwt(path_shp: str | Path, name: str, resolution: float = 1.0):
#     """ 计算CWT和WCWT值

#         path_shp: 包含后向轨迹线的shp文件路径
#         name: 浓度列名

#     2024-09-02 v1
#     """

#     # 读取shp文件
#     gdf_lines = gpd.read_file(path_shp, engine='pyogrio')

#     # 替换浓度列的-9999为nan
#     gdf_lines[name] = gdf_lines[name].replace(-9999, np.nan)

#     # 合并所有LineString到单一的MultiLineString
#     combined = MultiLineString(gdf_lines.geometry.tolist())

#     # 计算边界坐标
#     lon_west, lat_south, lon_east, lat_north = combined.bounds

#     # 生成经纬度数组
#     arr1d_lon = np.arange(lon_west, lon_east, resolution)
#     arr1d_lat = np.arange(lat_north, lat_south, -resolution)

#     # 生成2维网格经纬度
#     array2d_lon_m, array2d_lat_m = np.meshgrid(arr1d_lon, arr1d_lat)

#     # 生成2维网格多边形
#     array2d_polygons = create_grid_polygons(array2d_lat_m, array2d_lon_m)
#     print('shape:', array2d_polygons.shape)

#     # 将gdflines转换为pygeos对象
#     lines_geos = pygeos.from_shapely(gdf_lines.geometry)

#     # 数据存入字典
#     dict_result = {}

#     # ID写入字典
#     dict_result['ID'] = np.arange(1, array2d_polygons.size + 1)

#     list_var = ['Nij', 'CWT', 'WCWT']
#     for i in list_var:
#         dict_result[i] = []

#     # 遍历多边形
#     for polygon in array2d_polygons.flatten():

#         # 将polygon转换为pygeos对象
#         polygon_geos = pygeos.from_shapely(polygon)

#         # 取交集
#         bool_intersect = pygeos.intersects(lines_geos, polygon_geos)
        
#         # 计算有多少条轨迹经过该多边形，Nij
#         Nij = bool_intersect.sum()
#         if Nij == 0:
#             for i in list_var:
#                 dict_result[i].append(np.nan)
#             continue

#         # 取出相交的线
#         lines_intersect = lines_geos[bool_intersect]

#         # 计算有多少点位于多边形内
#         arr1d_points_num = np.array([countPointInPolygon(line, polygon_geos) for line in lines_intersect])

#         # 如果和为0
#         if np.sum(arr1d_points_num) == 0:
#             dict_result['Nij'].append(np.nan)
#             dict_result['CWT'].append(np.nan)
#             dict_result['WCWT'].append(np.nan)
#             continue

#         # 计算浓度与停留时间的乘积
#         arr1d_conc_time = gdf_lines[bool_intersect][name].values * arr1d_points_num

#         # 计算CWT值
#         CWTij = np.nansum(arr1d_conc_time) / np.sum(arr1d_points_num)

#         # 计算WPSCF值
#         WCWTij = CWTij * weight(Nij)

#         # 将PSCF值和WPSCF值存入列表
#         dict_result['Nij'].append(Nij)
#         dict_result['CWT'].append(CWTij)
#         dict_result['WCWT'].append(WCWTij)

#     # 将数据存入GeoDataFrame
#     gdf_cwt = gpd.GeoDataFrame(
#         data=dict_result, 
#         geometry=array2d_polygons.flatten(),
#         crs='EPSG:4326',
#         )
    
#     # 返回数据
#     return gdf_cwt


def wcwt(path_shp: str | Path, name: str, resolution: float = 1.0, extent: tuple[float, float, float, float] = None, path_csv: str | Path = None):
    """ 计算CWT和WCWT值

        path_shp: 包含后向轨迹线的shp文件路径
        name: 浓度列名
        resolution: 网格分辨率，°
        extent: 网格范围，(lon_west, lat_south, lon_east, lat_north)
        path_csv: 保存结果的csv文件路径

    2024-09-04 v2 用shapely库替换pygeos库，添加extent参数，添加保存csv文件功能
    """

    # 读取shp文件
    gdf_lines = gpd.read_file(path_shp, engine='pyogrio')

    # 替换浓度列的-9999为nan
    gdf_lines[name] = gdf_lines[name].replace(-9999, np.nan)

    if extent is None:
        # 合并所有LineString到单一的MultiLineString
        combined = MultiLineString(gdf_lines.geometry.tolist())

        # 计算边界坐标
        lon_west, lat_south, lon_east, lat_north = combined.bounds
    
    else:
        lon_west, lat_south, lon_east, lat_north = extent

    # 生成经纬度数组
    arr1d_lon = np.arange(lon_west, lon_east + resolution, resolution)
    arr1d_lat = np.arange(lat_north, lat_south - resolution, -resolution)

    # 生成2维网格经纬度
    array2d_lon_m, array2d_lat_m = np.meshgrid(arr1d_lon, arr1d_lat)

    # 生成2维网格多边形
    array2d_polygons = create_grid_polygons(array2d_lat_m, array2d_lon_m)
    # print('shape:', array2d_polygons.shape)

    # 将gdflines转换为pygeos对象
    # lines_geos = pygeos.from_shapely(gdf_lines.geometry)

    # 数据存入字典
    dict_result = {}

    # ID写入字典
    dict_result['ID'] = np.arange(1, array2d_polygons.size + 1)

    list_var = ['Nij', 'CWT', 'WCWT']
    for i in list_var:
        dict_result[i] = []

    # 遍历多边形
    for polygon in array2d_polygons.flatten():

        # 将polygon转换为pygeos对象
        # polygon_geos = pygeos.from_shapely(polygon)

        # 取交集
        # bool_intersect = pygeos.intersects(lines_geos, polygon_geos)
        bool_intersect = gdf_lines.geometry.intersects(polygon).values
        # print('bool_intersect:', bool_intersect)
        
        # 计算有多少条轨迹经过该多边形，Nij
        Nij = bool_intersect.sum()
        if Nij == 0:
            for i in list_var:
                dict_result[i].append(np.nan)
            continue

        # 取出相交的线
        # lines_intersect = lines_geos[bool_intersect]
        # lines_intersect = gdf_lines.geometry[bool_intersect]

        # 计算有多少点位于多边形内
        arr1d_points_num = np.array([countPointInPolygon(line, polygon) for line in gdf_lines.geometry[bool_intersect]])
        # arr1d_points_num = np.array([countPointInPolygon(line, polygon_geos) for line in lines_intersect])

        # 如果和为0
        if np.sum(arr1d_points_num) == 0:
            dict_result['Nij'].append(np.nan)
            dict_result['CWT'].append(np.nan)
            dict_result['WCWT'].append(np.nan)
            continue

        # 计算浓度与停留时间的乘积
        arr1d_conc_time = gdf_lines[bool_intersect][name].values * arr1d_points_num

        # 计算CWT值
        CWTij = np.nansum(arr1d_conc_time) / np.sum(arr1d_points_num)

        # 计算WPSCF值
        WCWTij = CWTij * weight(Nij)

        # 将PSCF值和WPSCF值存入列表
        dict_result['Nij'].append(Nij)
        dict_result['CWT'].append(CWTij)
        dict_result['WCWT'].append(WCWTij)

    # 将数据存入GeoDataFrame
    gdf_cwt = gpd.GeoDataFrame(
        data=dict_result, 
        geometry=array2d_polygons.flatten(),
        crs='EPSG:4326',
        )
        
    if path_csv is not None:

        # 提取WCWT列数据
        arr2d_wcwt = gdf_cwt['WCWT'].values.reshape(arr1d_lat.shape[0] - 1, arr1d_lon.shape[0] - 1)

        # 保存为csv文件
        np.savetxt(path_csv, arr2d_wcwt, delimiter=',')

    # 返回数据
    return gdf_cwt


""" 后面的函数待优化/删除 """
def get_nearest_gkd_coordinates(lat: float, lon: float, dict_code2coordinate, radius):
    """ 获取距离目标坐标点A小于radius km的国控站点的代码、坐标和距离

        lat：纬度；
        lon：经度；
        dict_code2coordinate： 字典，key=站点代码，value=(latitude, longitude)；
        radius：搜索半径，单位km；

        return: {'1001A': {'coord': [纬度, 经度], 'dist': 距离},
                 '1002A': {'coord': [纬度, 经度], 'dist': 距离},
                 ...
                }

    2022-12-30    v2

    待删除，使用NearestCoords代替
    """

    # t0 = time.time()
    """ 准备国控站点经纬度数组array2d_coord，第1列为纬度，第2列为经度 """
    list_code = [i for i in dict_code2coordinate.keys()]
    array2d_coord = np.array([dict_code2coordinate[i] for i in list_code])
    # print('array2d_coord:', array2d_coord)

    """ 筛选国控点中距离坐标点A经纬度相差小于1°的坐标数组 """
    # 将所有国控点坐标与坐标A的经纬度相减，
    # array2d_subtract中第1列为纬度相减的绝对值，第2列为经度相减的绝对值
    array2d_subtract = np.abs(array2d_coord - np.array([lat, lon]))
    # print('array2d_subtract:', array2d_subtract)

    # 符合条件的索引（纬度相差1°，距离约111km，这里用radius/100扩大筛选范围；60°纬线经度相差1°，距离约55km）
    index_ = np.where((array2d_subtract[:, 0] <= radius/100) & (array2d_subtract[:, 1] <= radius/50))

    # 符合条件的站点坐标
    array2d_coord = array2d_coord[index_]

    # 符合条件的站点代码
    array1d_code = [list_code[i] for i in index_[0]]
    # print('array1d_code:', array1d_code)
    if not array1d_code:
        return {}

    # t1 = time.time()

    """ 计算距离 """
    array1d_lat1 = array2d_coord[..., 0]
    array1d_lon1 = array2d_coord[..., 1]

    # 准备目标坐标点A的经纬度数组
    array1d_lat2 = np.ones(shape=array1d_lat1.shape) * lat
    array1d_lon2 = np.ones(shape=array1d_lat1.shape) * lon

    # 计算坐标点A与每个国控站点的距离
    array1d_distance = geodistances(array1d_lat1=array1d_lat1,
                                          array1d_lon1=array1d_lon1,
                                          array1d_lat2=array1d_lat2,
                                          array1d_lon2=array1d_lon2,
                                          )
    # print('array1d_distance:', array1d_distance)

    # 筛选半径radius范围内的坐标索引
    index_lessthan_radius = np.where(np.array(array1d_distance) <= radius)[0]
    # print('index_lessthan_radius:', index_lessthan_radius)

    # 准备返回数据
    dict_result = {}
    for i in index_lessthan_radius:
        code = array1d_code[i]
        coord = array2d_coord[i]
        dist = array1d_distance[i]
        dict_result[code] = {'coord': coord, 'dist': dist}

    return dict_result


def read_gkd_sites_vars_dt(dir_h5: os.PathLike, sites: list, dt: pd.Timestamp, vars: list):
    """ 读取国控点hdf5数据中多个站点多个物种单个时间点的数据均值 
    
        dir_h5: str, hdf5文件所在目录
        sites: list, 站点代码列表
        dt: pd.Timestamp, 时间点
        vars: list, 物种列表，如：['PM2.5', 'PM10', 'SO2', 'NO2', 'O3', 'CO']

        return: list

    单进程多线程
    2024-08-02 v1
    """

    # 线程池初始化
    pool = ThreadPoolExecutor(max_workers=len(vars))

    # 准备参数
    args_ = [(os.path.join(dir_h5, f'{var}.h5'), sites, dt) for var in vars]

    # 提交任务
    list_result = list(pool.map(read_gkd_sites_var_dt, *zip(*args_)))

    # 提取结果
    # dict_result = dict(zip(vars, list_result))
    # list_result = [i for i in list_result]

    # 返回数据
    return list_result


def read_gkd_sites_var_dt(path_h5: os.PathLike, sites: list, dt: pd.Timestamp):
    """ 读取国控点hdf5数据中多个站点单物种单个时间点的数据均值

        path_h5: str, hdf5文件路径
        sites: list, 站点代码列表
        dt: pd.Timestamp, 时间点

        return: float

    单进程
    2024-08-02 v1
    """

    # 打开hdf5文件
    f = h5py.File(name=path_h5, mode="r")

    # 数据集
    dset_ = f[str(dt.year)]

    # 时间索引
    index_ = dset_.attrs['datetime']

    # hdf5文件中的columns
    columns_ = dset_.attrs['columns']

    # dt在index_中的位置索引
    index_dt = np.where(index_ == dt.timestamp())[0]
    
    # 判断index_dt是否为空
    if len(index_dt) == 0:
        f.close()
        return np.nan
    
    # print('index_dt:\n', np.where(index_ == dt.timestamp())[0])
    # index_dt = np.where(index_ == dt.timestamp())[0][0]

    # 站点代码列表在columns中的位置索引
    index_column = np.where(np.isin(columns_, sites))[0]

    # 取值
    array1d_data = dset_[index_dt[0], index_column]

    # 判断是否为nan
    if np.isnan(array1d_data).all():
        f.close()
        return np.nan
    
    # 计算均值
    mean_ = np.nanmean(array1d_data)

    # 关闭hdf5文件
    f.close()

    # 返回数据
    return mean_


if __name__ == '__main__':

    # 示例
    latlon1 = [40, 110]
    latlon2 = [30, 120]

    # 生成一个2维数组，第一列为纬度，第二列为经度
    arr2d_coords = np.full(shape=(10000, 2), fill_value=latlon2)
    # print(arr2d_coords)
    # exit(0)

    t0 = time.time()
    
    dist1 = geodistances(latlon1=latlon1, latlons=arr2d_coords)

    t1 = time.time()
    geod = Geod(ellps='WGS84')
    for i in range(10000):
        # 计算距离
        dist2 = geod.inv(lons1=latlon1[1], lats1=latlon1[0], lons2=latlon2[1], lats2=latlon2[0])[2] * 0.001

    t2 = time.time()


    # print('距离：', dist1, dist2)
    print('time:', t1 - t0, t2-t1)
