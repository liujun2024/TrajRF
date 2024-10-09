""" Calculate the average distance between upwind sites and receptor site (1410A here) """

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _traj as traj
import _gkd as gkd
import _utils as utils


if __name__ == '__main__':

    # upwind cities to be analyzed
    list_city = ['湛江市', '茂名市', '阳江市', '江门市', '广州市', 
                 '深圳市', '珠海市', '佛山市', '中山市', '东莞市', 
                 '惠州市', '肇庆市', '汕尾市', '汕头市', '揭阳市', 
                 '潮州市', '漳州市', '厦门市']

    # path of hdf5 file
    path_h5 = Path(r'traj.h5')

    # site information
    si = gkd.SiteInfo(path_site_list=r'NewSiteList.csv')

    # the code of receptor site
    sitecode = '1410A'

    # the coordinate of receptor site
    coord = si.dict_code2coordinate[sitecode]

    # convert coordinate to string
    str_coord = ','.join([str(traj.round_accurately(i, 3)) for i in si.dict_code2coordinate[sitecode]])
    print('str_coord:', sitecode, coord, str_coord)
    
    # read traj from hdf5 file
    dict_traj = traj.read_traj_from_h5(
        path_h5=path_h5,
        list_coord=[coord],
        list_length=range(-36, 0, 1),
    )[str_coord]

    # merge traj
    list_arr2d = []
    for hour, arr2d_traj in dict_traj.items():

        # Slice to get latitude and longitude
        arr2d_latlon = arr2d_traj[:, :2] * 0.001

        # merge hour to arr2d_latlon
        arr2d_latlonhour = np.hstack((arr2d_latlon, np.full(shape=(arr2d_latlon.shape[0], 1), fill_value=hour)))
  
        # append to list
        list_arr2d.append(arr2d_latlonhour)

    # joint all the 2d array in list_array2d_hour
    arr2d_jointed = np.vstack(list_arr2d)

    # site codes for all sites within the upwind cities
    list_sitecode = []
    for c in list_city:
        list_sitecode += si.dict_city2code[c].tolist()

    # Filter the trajectory points within a 5 km radius of the upwind sites
    # and calculate the average travel time of these points to the receptor site (1410A here)"
    dict_dist = dict()

    # iterate over all sites
    for site in list_sitecode:
        
        # coordinates of the site
        lat_site, lon_site = si.dict_code2coordinate[site]

        # Calculate the spherical distance between the site and all trajectory points
        distance_site = utils.geodistances(latlon1=(lat_site, lon_site), latlons=arr2d_jointed[:, :2])

        # merge distance to arr2d_jointed
        arr2d_latlonhourdis = np.hstack((arr2d_jointed, np.array(distance_site).reshape(-1, 1)))

        # get the boolean row index for filtering trajectory points within 5 km of the site
        index_filter = arr2d_latlonhourdis[:, -1] <= 5

        # skip if index_filter is all False
        if not np.any(index_filter):
            continue
        
        # get the hour column
        arr1d_hour = arr2d_latlonhourdis[index_filter, 2]

        # Calculate the mean and standard deviation
        mean_hour = np.nanmean(arr1d_hour)
        std_hour = np.nanstd(arr1d_hour)

        # save to dictionary
        dict_dist[site] = (mean_hour, std_hour)
        print(f'site code: {site}, mean_hour: {mean_hour}, std_hour: {std_hour}')

    # Calculate the city-level average of air mass travel time between upwind sites and the receptor site
    dict_city = dict()

    # iterate over all cities
    for city in list_city:

        # travel hour for all sites within the city
        list_hour_mean = [dict_dist[site][0] for site in si.dict_city2code[city] if site in dict_dist.keys()]
        list_hour_std = [dict_dist[site][1] for site in si.dict_city2code[city] if site in dict_dist.keys()]

        # Calculate the mean and standard deviation in city-level
        mean_hour = np.nanmean(list_hour_mean)
        std_hour = np.nanmean(list_hour_std)

        # save to dictionary
        dict_city[city] = (mean_hour, std_hour)
    
    # convert dictionary to pandas.DataFrame
    df_dist = pd.DataFrame(dict_city, index=['hour_mean', 'hour_std']).T
    
    # Calculate the abolute value of the mean travel time
    df_dist = df_dist.abs()

    # sort by mean travel time
    df_dist.sort_values(by='hour_mean', ascending=True, inplace=True)

    # rename chinese city name to english
    df_dist.rename(
        index={
            '湛江市': 'Zhanjiang(ZJ)', 
            '茂名市':'Maoming(MM)',
            '阳江市':'Yangjiang(YJ)',
            '江门市':'Jiangmen(JM)',
            '广州市': 'Guangzhou(GZ)',
            '深圳市':'Shenzhen(SZ)',
            '珠海市':'Zhuhai(ZH)',
            '佛山市':'Foshan(FS)',
            '中山市':'Zhongshan(ZS)',
            '东莞市':'Dongguan(DG)',
            '惠州市':'Huizhou(HZ)',
            '肇庆市':'Zhaoqing(ZQ)',
            '汕尾市': 'Shanwei(SW)', 
            '汕头市':'Shantou(ST)',
            '潮州市':'Chaozhou(CZ)',
            '揭阳市':'Jieyang(JY)',
            '漳州市':'Zhangzhou(ZZ)',
            '厦门市':'Xiamen(XM)',
            }, 
        inplace=True)

    # save to csv file
    df_dist.to_csv('distance.csv', index=True, encoding='gbk')
    
    # print the result
    print('df_dist : \n', df_dist)
