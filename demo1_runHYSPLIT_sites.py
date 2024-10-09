""" Generate HYSPLIT trajectories for single or multiple sites """

from __future__ import annotations 
import numpy as np
from pathlib import Path
import _gkd as gkd
import _traj as traj


# Working directory for HYSPLIT 
dir_working = Path(__file__).parent / 'working'

# Path of the HYSPLIT executable file (hyts_std.exe)
path_exe = dir_working / 'hyts_std.exe'

# Directory for meteorological data (gdas1 here) that can 
# be downloaded from https://www.ready.noaa.gov/data/archives/gdas1/
dir_gdas1 = Path(r'E:\bigdata\hysplit\gdas1')

# Directory for saving trajectories
dir_traj = dir_working.parent / 'traj'
if not dir_traj.exists():
    dir_traj.mkdir(parents=True)


if __name__ == '__main__':

    # site code list, here 1410A belongs to Haikou in Hainan province
    list_sitecode = ['1410A']

    # site information
    si = gkd.SiteInfo(path_site_list=r'NewSiteList.csv')

    # Preparing the dictionary of site coordinates
    dict_code2coords = {}

    for sitecode in list_sitecode:
    
        # get the coordinates of site
        coord_receptor = si.dict_code2coordinate[sitecode]

        # skip when coordinate is nan
        if np.any(np.isnan(coord_receptor)):
            continue

        # save the coordinates
        dict_code2coords[sitecode] = coord_receptor

    """ Run HYSPLIT """
    traj.RunHYSPLIT(
        dir_save=dir_traj,
        dict_coords=dict_code2coords,
        m_agl=100,
        runtime=-36,
        datetime=['2021-10-01 00:00:00', '2022-02-28 23:00:00'],
        dir_working=dir_working,
        dir_meteo=dir_gdas1,
        path_exe=path_exe,
        ).get_traj()
