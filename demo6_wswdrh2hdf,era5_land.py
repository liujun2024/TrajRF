""" Calculate 10m wind speed (ws10) and wind direction (wd10) from u10, v10.
    Calculate relative humidity (rh) from t2m, d2m.
    Save to hdf5 file """

from __future__ import annotations
from pathlib import Path
import pandas as pd
import _era5 as era5
import _hdf5 as hdf5


if __name__ == "__main__":

    # related variables 
    list_var = ['u10', 'v10', 'd2m', 't2m']

    # directory to save the hdf5 files
    dir_hdf5 = Path(__file__).parent / 'ERA5' / 'hdf5-era5-land'

    # path of saving ws10, wd10, rh
    path_ws10 = dir_hdf5 / 'ws10.h5'
    path_wd10 = dir_hdf5 / 'wd10.h5'
    path_rh = dir_hdf5 / 'rh.h5'

    # delete the existing h5 files
    if path_ws10.exists():
        path_ws10.unlink()
    if path_wd10.exists():
        path_wd10.unlink()
    if path_rh.exists():
        path_rh.unlink()

    # reading u10, v10, d2m, t2m from hdf5 files
    dict_data = hdf5.read_hdf5_by_sites(
        dir_h5=dir_hdf5, 
        species=list_var, 
        date_range=['2000-01-01', '2024-12-31'], sites=None)
    
    # datetime index of u10 and v10
    dt_u10 = dict_data['u10'].index
    dt_v10 = dict_data['v10'].index

    # get insection of indexs
    dt_wind = dt_u10.intersection(dt_v10)

    # reindex u10 and v10
    df_u10 = dict_data['u10'].loc[dt_wind, :]
    df_v10 = dict_data['v10'].loc[dt_wind, :]

    # calculate ws10 and wd10
    ws, wd = era5.cal_wind(u=df_u10.to_numpy(), v=df_v10.to_numpy())
    df_ws = pd.DataFrame(data=ws, index=dt_wind, columns=df_u10.columns)
    df_wd = pd.DataFrame(data=wd, index=dt_wind, columns=df_u10.columns)

    # datetime index of t2m and d2m
    dt_t2m = dict_data['t2m'].index
    dt_d2m = dict_data['d2m'].index

    # get insection of indexs
    dt_rh = dt_t2m.intersection(dt_d2m)

    # reindex t2m and d2m
    df_t2m = dict_data['t2m'].loc[dt_rh, :]
    df_d2m = dict_data['d2m'].loc[dt_rh, :]

    # calculate rh
    rh = era5.cal_rh(t=df_t2m.to_numpy(), dp=df_d2m.to_numpy(), unit='C')
    df_rh = pd.DataFrame(data=rh, index=dt_rh, columns=df_t2m.columns)

    # save ws10 to hdf5 file by year
    for dt, df_year in df_ws.groupby(df_ws.index.year):
        hdf5.df2h5_era5(
            data_=df_year,
            path_hdf5=path_ws10,
            short_name='ws10',
            long_name='10 metre wind speed',
            units='m s**-1',
            )

    #  save wd10 to hdf5 file by year
    for dt, df_year in df_wd.groupby(df_wd.index.year):
        hdf5.df2h5_era5(
            data_=df_year,
            path_hdf5=path_wd10, 
            short_name='wd10', 
            long_name='10 metre wind direction',
            units='angles(°, 0°: wind from the north, 90°: wind from the east)',
            )

    # save rh file by year
    for dt, df_year in df_rh.groupby(df_rh.index.year):
        hdf5.df2h5_era5(
            data_=df_year, 
            path_hdf5=path_rh, 
            short_name='rh', 
            long_name='relative humidity',
            units='%',
            )
