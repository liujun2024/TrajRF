""" get site data from the HDF5 file and save it as a CSV file. """

from __future__ import annotations
from pathlib import Path
from PyQt6.QtWidgets import QFileDialog, QApplication
import os
import sys
import pandas as pd
import _gkd as gkd
import _era5 as era5


# datetime range to read
dt_start, dt_end = ['2015-01-01', '2024-01-30']

# variables to read
list_var_land = ['u10', 'v10', 't2m', 'ws10', 'wd10', 'rh', 'sp', 'tp', 'ssrd', 'strd', 'd2m']
list_var_single_levels = ['u100', 'v100', 'ws100', 'wd100', 'blh']
list_var = list_var_land + list_var_single_levels

# directory of hdf5 files
dir_hdf5_land = Path(__file__).parent / 'ERA5' / 'hdf5-era5-land'
dir_hdf5_single_levels = Path(__file__).parent / 'ERA5' / 'hdf5-era5-single-levels'

# site information
si = gkd.SiteInfo(path_site_list=r'NewSiteList.csv')

# site list to read
list_sites = ['1410A']

if __name__ == "__main__":

    app = QApplication(sys.argv)

    # process pool for reading hdf5 files
    thread_get_era5 = era5.ReadEra5Site(
        dir_h5_land=dir_hdf5_land,
        dir_h5_single_levels=dir_hdf5_single_levels,
        species_land=list_var_land,
        speices_single_levels=list_var_single_levels,
        date_range=[dt_start, dt_end],
        dict_percentile={},
        site_info=si,
        list_sites=list_sites,
        time_resolution='H',
        )
    
    # launch thread
    thread_get_era5.start()
    thread_get_era5.wait()
    app.quit()

    # get data of all variables and sites
    dict_data_era5 = thread_get_era5.data

    # classify data by city
    dict_site = dict()
    for site in list_sites:

        # get site data of all variables
        list_df = [dict_data_era5[v][site] for v in list_var if site in dict_data_era5[v].keys()]
        if len(list_df) != len(list_var):
            continue

        # concatenate data of all variables to one pd.DataFrame
        df_site = pd.concat(objs=list_df, axis=1)

        # columns
        df_site.columns = list_var

        # save to dictionary
        dict_site[site] = df_site

    # open file dialog to select save directory
    dir_save = QFileDialog.getExistingDirectory(None, 'Select Save Directory', r'./')
    if not dir_save:
        exit(0)

    # save data
    for site, df_site in dict_site.items():

        # delete rows with NaN
        df_site.dropna(how='any', axis=0, inplace=True)

        # path to save
        path_site = os.path.join(dir_save, site + '.csv')

        # save
        df_site.to_csv(path_site)
