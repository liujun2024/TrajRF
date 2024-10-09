""" Read the raw ERA5 files (*.nc) and save the data for specific sites into an HDF5 file, 
    suitable for processing ERA5 single levels data """

from __future__ import annotations
from pathlib import Path
from itertools import chain
import pandas as pd
import _gkd as gkd
import _netcdf as nc
import _hdf5 as hdf5


# target variables in ERA5 single levels
dict_var_single_levels = {
    'blh': 'boundary_layer_height',
    'u100': '100m_u_component_of_wind',
    'v100': '100m_v_component_of_wind',
}

# directory of raw ERA5 single levels data, the directory structure is as follows:
# 
# reanalysis-era5-single-levels
# ├── boundary_layer_height
# │   ├── 202001.nc
# │   ├── 202002.nc
# │   └── ...
# ├── ...

dir_nc = Path(__file__).parent / 'ERA5' / 'reanalysis-era5-single-levels'

# directory to save the hdf5 files
dir_hdf5 = Path(__file__).parent / 'ERA5' / 'hdf5-era5-single-levels'

# create directory if not exists
if not dir_hdf5.exists():
    dir_hdf5.mkdir(parents=True)


if __name__ == '__main__':

    # site information
    si = gkd.SiteInfo(path_site_list=r'NewSiteList.csv')

    # all cities in China
    list_city = si.list_city_china

    # all site codes in China
    list_code = list(chain(*[si.dict_city2code[i] for i in list_city]))

    # coordinates of all sites in China
    lat_lon = [si.dict_code2coordinate[i] for i in list_code]

    # save coordinates to pd.DataFrame
    df_site = pd.DataFrame(data=lat_lon)
    df_site.index = list_code
    df_site.index.name = 'site'
    df_site.columns = ['lat', 'lon']

    # delete row with NaN
    df_site.dropna(how='any', axis=0, inplace=True)

    # read ERA5 data
    for short_name, long_name in dict_var_single_levels.items():

        # directory of current variable
        dir_v = dir_nc / long_name

        # skip if directory does not exist
        if not dir_v.exists():
            continue

        # path of saving hdf5 file
        path_hdf5 = dir_hdf5 / f'{short_name}.h5'

        # all *.nc files in directory
        list_files = dir_v.glob('*.nc')

        if not list_files:
            continue

        # year
        list_year = sorted(list(set([i[:4] for i in list_files])), reverse=False)

        # iterate over years
        for year in list_year:

            # all files of current year
            list_files_year = [i for i in list_files if i.stem.startswith(year)]

            # read data
            dict_result_year = nc.read_era5_without_levels_by_coord(list_files=list_files_year, coord=df_site)

            # save to hdf5 file
            hdf5.df2h5_era5(
                data_=dict_result_year['data'], 
                path_hdf5=path_hdf5, 
                short_name=dict_result_year['short_name'], 
                long_name=dict_result_year['long_name'], 
                units=dict_result_year['units'],
                )
