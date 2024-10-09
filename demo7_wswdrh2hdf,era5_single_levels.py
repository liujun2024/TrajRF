""" Calculate 100m wind speed (ws100) and wind direction (wd100) from u100, v100.
    Save to hdf5 file """

from __future__ import annotations
from pathlib import Path
import pandas as pd
import _era5 as era5
import _hdf5 as hdf5


if __name__ == "__main__":

    # 需要读取的变量名列表
    list_var = ['u100', 'v100']

    # directory to save the hdf5 files
    dir_hdf5 = Path(__file__).parent / 'ERA5' / 'hdf5-era5-single-levels'

    # path of saving ws100, wd100
    path_ws100 = dir_hdf5 / 'ws100.h5'
    path_wd100 = dir_hdf5 / 'wd100.h5'

    # delete the existing h5 files
    if path_ws100.exists():
        path_ws100.unlink()
    if path_wd100.exists():
        path_wd100.unlink()

    # reading u100, v100 from hdf5 files
    dict_data = hdf5.read_hdf5_by_sites(
        dir_h5=dir_hdf5, 
        species=list_var, 
        date_range=['2000-01-01', '2024-12-31'], sites=None)
    
    # datetime index of u100 and v100
    dt_u100 = dict_data['u100'].index
    dt_v100 = dict_data['v100'].index

    # get insection of indexs
    dt_wind = dt_u100.intersection(dt_v100)

    # reindex u100 and v100
    df_u100 = dict_data['u100'].loc[dt_wind, :]
    df_v100 = dict_data['v100'].loc[dt_wind, :]

    # calculate ws100 and wd100
    ws, wd = era5.cal_wind(u=df_u100.to_numpy(), v=df_v100.to_numpy())
    df_ws = pd.DataFrame(data=ws, index=dt_wind, columns=df_u100.columns)
    df_wd = pd.DataFrame(data=wd, index=dt_wind, columns=df_u100.columns)

    # save ws100 to hdf5 file by year
    for dt, df_year in df_ws.groupby(df_ws.index.year):
        hdf5.df2h5_era5(
            data_=df_year,
            path_hdf5=path_ws100,
            short_name='ws100',
            long_name='100 metre wind speed',
            units='m s**-1',
            )

    #  save wd100 to hdf5 file by year
    for dt, df_year in df_wd.groupby(df_wd.index.year):
        hdf5.df2h5_era5(
            data_=df_year,
            path_hdf5=path_wd100, 
            short_name='wd100', 
            long_name='100 metre wind direction',
            units='angles(°, 0°: wind from the north, 90°: wind from the east)',
            )
