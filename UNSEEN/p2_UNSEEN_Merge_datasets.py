import os
import sys

import src.cdsretrieve as retrieve
import src.preprocess as preprocess

import numpy as np
import xarray as xr
import pandas as pd

model_folder = 'Global_forecasts'#'Global_forecasts' or 'Global_hindcasts'
base_directory=f'../../data/{model_folder}/'
output_directory=f'../../model_runs/SEAS5_statistics/independence_tests/{model_folder}/'



sliced_datasets = [] 
# Merge SEAS5
for month_number in range(1, 13):
#for month_number in range(1, 2):
    month_directory = base_directory + f"month{month_number}/"
    print('month_directory:', month_directory)
    print('month_number:', month_number)
    
    # create output directory
    output_month_directory = output_directory + f"month{month_number}/"
    if not os.path.exists(output_month_directory):
        os.makedirs(output_month_directory)

    #SEAS5 = preprocess.merge_SEAS5(folder = month_directory + "SEAS5/", target_months = [month_number], new_cds=True)
    
    if model_folder == 'Global_forecasts':
        SEAS5 = preprocess.merge_SEAS5(folder = month_directory + "SEAS5/", target_months = [month_number], new_cds=False)
        start_date = '2017-01-01' #'2020-01-01'
        end_date = '2023-12-31'
        SEAS5_sliced = SEAS5.sel(time=slice(start_date, end_date))
        print('SEAS5_sliced:', SEAS5_sliced)
    else:
        SEAS5 = preprocess.merge_SEAS5(folder = month_directory + "SEAS5/", target_months = [month_number], new_cds=True)
        SEAS5_sliced = SEAS5
        print('SEAS5_sliced:', SEAS5_sliced)
    

    encoding = {"msl": {"dtype": "float32", "_FillValue": -9999}}
    
    sliced_datasets.append(SEAS5_sliced)
# Concatenate all the datasets along the 'time' dimension
combined_sliced = xr.concat(sliced_datasets, dim='time')

# Define encoding for saving
encoding = {"msl": {"dtype": "float32", "_FillValue": -9999}}

# Save the concatenated dataset to a single NetCDF file
combined_sliced.to_netcdf(output_directory + 'SEAS5_merged.nc', encoding=encoding)


# Merge ERA5
ERA5_timeseries = xr.open_mfdataset(base_directory + "*/ERA5/*.nc", combine='by_coords')
ERA5_monthly_min = ERA5_timeseries.resample(time="ME").min()
print('ERA5 monthly min:', ERA5_monthly_min.msl)
print('ERA5 monthly min time:', ERA5_monthly_min.time.values)

ERA5_monthly_min.to_netcdf(output_directory + 'ERA5.nc')
  

