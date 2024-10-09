""" Save trajectories to HDF5 file """

from __future__ import annotations

from pathlib import Path
import _traj as traj


if __name__ == '__main__':

    # directory of trajectory files
    dir_traj = Path(__file__).parent / 'traj'

    # path of hdf5 files
    path_hdf5 = dir_traj.parent / 'traj.h5'

    #  read the trajectory files and save to hdf5 file
    traj.traj2h5(dir_traj=dir_traj, path_h5=path_hdf5)
