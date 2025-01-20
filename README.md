# Traj-RF

This is a Random Forest Regression model that uses air mass trajectories as input features, along with pollutant concentrations and meteorological parameters. The trajectory data is from calling HYSPLIT by Python, pollutant concentration data from national control stations, and meteorological parameters from ERA5 reanalysis data.

## 国控点污染物浓度 Pollutant concentration
The raw pollutant concentration data comes from the China National Environmental Monitoring Centre (CNEMC) (https://air.cnemc.cn:18007/), with thanks to Wang Xiaolei for compiling and releasing the historical datasets (https://quotsoft.net/air/). Here, we use our previously developed Python-based tool (https://gitee.com/liujun2023/gkd) for parsing pollutant concentration data.

## 轨迹信息 Trajectory (HYSPLIT)
The necessary files to generate the trajectories were separated from the HYSPLIT program and stored in the working folder in root directory. Batch generation of the trajectories can then be completed using our custom Python scripts.

1. demo1: Generate HYSPLIT trajectories for single or multiple sites.
2. demo2: Save trajectories to HDF5 file.
3. demo3: Calculate the average distance between upwind sites and receptor site.

## 气象因素 Meteorological parameters (ERA5)
The raw meteorological parameters is from ERA5 reanalysis dataset published by European Centre for Medium-Range Weather Forecasts (ECMWF). It can be downloaded from [ERA5-Land hourly data from 1950 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview) and [ERA5 hourly data on single levels from 1940 to present](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview). When downloading, the ERA5-Land dataset saves one day's data per file, while the ERA5-Single-Levels dataset saves one month's data per file, allowing for streamlined processing using the subsequent Python scripts.
1. demo4: Read the raw ERA5-Land datasets ('yyyymmdd.nc') and save the data for specific sites into an HDF5 file.
2. demo5: Read the raw ERA5-single-levels datasets ('yyyymm.nc') and save the data for specific sites into an HDF5 file.
3. demo6: Calculate 10m wind speed (ws10) and wind direction (wd10) from u10, v10 of ERA5-Land datasets and calculate relative humidity (rh) from t2m, d2m of ERA5-Land datasets, then save to hdf5 file.
4. demo7: Calculate 100m wind speed (ws100) and wind direction (wd100) from u100, v100 of ERA5-single-levels datasets and save to hdf5 file.
5. demo8: Get hourly meteorological data for specific sites from the HDF5 file and save it as a CSV file.

## 训练模型 Training machine learning model
Merge local pollutant concentrations, meteorological data, and pollutant concentration data from upwind cities with time delayed into a CSV file. Ensure the first column is the datetime, the last column is the target variable, and the remaining columns are the independent variables. Save this CSV file in the 'ML' folder in the root directory. Running the following script will allow you to train, test, interpret the model results, and output the model performance and so on.
1. demo9: Read data from csv file and train and test Random Forest Regression model

> Instructions for running above scripts can be found within the file

## 如何引用 How to cite
Liu Jun, Chen Meiru, Chu Biwu, Chen Tianzeng, Ma Qingxin, Wang Yonghong, Zhang Peng, Li Hao, Zhao Bin, Xie Rongfu, Huang Qing, Wang Shuxiao, and He Hong. Assessing the Significance of Regional Transport in Ozone Pollution through Machine Learning: A Case Study of Hainan Island, ACS ES&T Air, 2025.
https://doi.org/10.1021/acsestair.4c00297


## 问题反馈 Feedback
进入项目的Issues[反馈](https://github.com/liujun2024/Traj-RF/issues)
