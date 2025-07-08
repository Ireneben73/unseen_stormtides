import sys
import xarray as xr
import numpy as np
import pandas as pd
from scipy.signal import detrend
import random
import matplotlib.pyplot as plt

station = int(sys.argv[1])

#--------------------------------------------------------------------------------------------------------------------
## 0. Functions
#--------------------------------------------------------------------------------------------------------------------
    
def pot(data, seas5, percentile=99.0):
    """
    Compute return levels using the Peaks Over Threshold (POT) method with a Generalized Pareto Distribution (GPD)
    and estimate confidence intervals using bootstrapping.

    Parameters:
    - data (xarray.DataArray): The time series data (ensemble_member, stations, time).
    - seas5 (bool): Whether to process data as an ensemble with multiple members.
    - percentile (float, optional): The percentile threshold for exceedance (default: 99.5).
    
    Returns:
    - combined_waterlevels: Declustered exceedances for all stations.
    """
    print('SEAS5:', seas5)
    def decluster_exceedances(exceedance_times, exceedance_values, time_diff_threshold=3):
        declustered_times = []
        declustered_values = []
        current_cluster_times = [exceedance_times[0]]
        current_cluster_values = [exceedance_values[0]]
        
        for i in range(1, len(exceedance_times)):
            time_diff = (exceedance_times[i] - exceedance_times[i - 1]).astype('timedelta64[D]').item().days
            if time_diff <= time_diff_threshold:
                current_cluster_times.append(exceedance_times[i])
                current_cluster_values.append(exceedance_values[i])
            else:
                max_idx = np.argmax(current_cluster_values)
                declustered_times.append(current_cluster_times[max_idx])
                declustered_values.append(current_cluster_values[max_idx])
                current_cluster_times = [exceedance_times[i]]
                current_cluster_values = [exceedance_values[i]]
        
        max_idx = np.argmax(current_cluster_values)
        declustered_times.append(current_cluster_times[max_idx])
        declustered_values.append(current_cluster_values[max_idx])
        
        return declustered_times, declustered_values

    # Compute the threshold across all ensemble members and time
    data = data.compute()
    threshold = np.percentile(data.values, percentile)

    # Function to process each station
    def process_station(station_data, threshold_value):
        # Exceedances: values above the threshold
        combined_waterlevels = []
        if seas5:
            print('SEAS5!')
            for ensemble in data.ensemble_member.values:
                data_ens = data.sel(ensemble_member=ensemble)
                exceedances = data_ens.where(data_ens > threshold, drop=True)
                exceedance_times = exceedances.time.values
                exceedance_values = exceedances.values
                _, declustered_waterlevels = decluster_exceedances(exceedance_times, exceedance_values)
                combined_waterlevels.extend(declustered_waterlevels)
        else:
            exceedances = station_data.where(station_data > threshold_value, drop=True)
            exceedance_times = exceedances.time.values
            exceedance_values = exceedances.values
            _, declustered_waterlevels = decluster_exceedances(exceedance_times, exceedance_values)
            combined_waterlevels.extend(declustered_waterlevels)
        
        return combined_waterlevels

    combined_waterlevels = process_station(data, threshold)
    #print('combined_waterlevels:', combined_waterlevels)
    return combined_waterlevels

#--------------------------------------------------------------------------------------------------------------------
## 1. Prepare input data
#--------------------------------------------------------------------------------------------------------------------

ensemble_members = range(0, 15)  # adjust as needed for your ensemble range
annual_max_seas5 = []
#detrended_seas5 = []
seas5 = []

# Loop through ensemble members
for ensemble_member in ensemble_members:
    gtsm_seas5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/Europe/gtsm_EU_*_{ensemble_member}.nc').isel(stations=station)
    print('gtsm_seas5:', gtsm_seas5)
    

    
    #gtsm_seas5_detrended['waterlevel'] = detrend(gtsm_seas5['waterlevel'])
    # Prepare data for POT 3-day declustering
    gtsm_daily_seas5 = gtsm_seas5.assign_coords(ensemble_member=ensemble_member)
    seas5.append(gtsm_daily_seas5)

# Concatenate along the new ensemble dimension
seas5_ensemble = xr.concat(seas5, dim="ensemble_member")
print('seas5_ensemble:', seas5_ensemble)    

#--------------------------------------------------------------------------------------------------------------------
## 3. ERA5
#--------------------------------------------------------------------------------------------------------------------

#gtsm_era5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/ERA5/Europe/gtsm_EU_*.nc').isel(stations=station).sel(time=slice(detrended_seas5_ensemble.time.min().values, detrended_seas5_ensemble.time.max().values))
gtsm_era5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/ERA5/Europe/gtsm_EU_*.nc').isel(stations=station).sel(time=slice(seas5_ensemble.time.min().values, seas5_ensemble.time.max().values))

#--------------------------------------------------------------------------------------------------------------------
## 4. POT for SEAS5
#--------------------------------------------------------------------------------------------------------------------

def pot_with_seas5(time_series):
    #print('timeseries:', time_series)
    #return time_series
    return pot(time_series, seas5=True)
    
def pot_with_era5(time_series):
    #print('timeseries:', time_series)
    #return time_series
    return pot(time_series, seas5=False)


n_bootstraps = 1000  # Number of bootstrap iterations

print('STATION:', station)

# SEAS5
#station_data_seas5 = detrended_seas5_ensemble['waterlevel']#.isel(stations=station)
station_data_seas5 = seas5_ensemble['waterlevel']#.isel(stations=station)
pot_result_seas5 = pot_with_seas5(station_data_seas5)

# ERA5
#station_data_era5 = gtsm_era5_detrended['waterlevel']#.isel(stations=station)
station_data_era5 = gtsm_era5['waterlevel']#.isel(stations=station)
pot_result_era5 = pot_with_era5(station_data_era5)

# Bootstrapping: Generate 1000 resamples of pot_result_seas5
n_samples = len(pot_result_era5)
if len(pot_result_seas5) > 0:  # Ensure there's data to sample from

    bootstrapped_seas5 = np.random.choice(pot_result_seas5, size=(n_bootstraps, n_samples), replace=True)
    bootstrap_means = np.mean(bootstrapped_seas5, axis=1)  # Faster computation
    # Compute 95% confidence intervals
    ci_lower = np.percentile(bootstrap_means, 2.5, axis=0)  # Lower bound
    ci_upper = np.percentile(bootstrap_means, 97.5, axis=0)  # Upper bound
    ci_median = np.percentile(bootstrap_means, 50.0, axis=0)  # Upper bound
else:
    bootstrap_means = np.array([])  # Empty case
    ci_lower, ci_upper, ci_median = np.nan, np.nan, np.nan  # Handle missing values

# Compute mean of ERA5 annual maxima for the station
mean_era5 = np.mean(pot_result_era5)  # ERA5 mean
abs_diff = np.abs(mean_era5 - ci_median)

threshold=0.1 # 10 cm

# Classification
if np.isnan(ci_lower) or np.isnan(ci_upper):
    status = "No Data"
elif (mean_era5 < ci_lower) & (abs_diff < threshold):
    status = "Below 95% CI Small Diff"
elif (mean_era5 < ci_lower) & (abs_diff >= threshold):
    status = "Below 95% CI Large Diff"
elif (mean_era5 > ci_upper) & (abs_diff < threshold):
    status = "Above 95% CI Small Diff"
elif (mean_era5 > ci_upper) & (abs_diff >= threshold):
    status = "Above 95% CI Large Diff"
else:
    status = "Within 95% CI"
print('status:', status)

# Ensure station locations exist
station_lats = np.atleast_1d(station_data_seas5['station_y_coordinate'].values)
station_lons = np.atleast_1d(station_data_seas5['station_x_coordinate'].values)

status_array = np.array([status], dtype=object)

# Create xarray Dataset
ds = xr.Dataset(
    data_vars={'status': ('station', status_array)},
    coords={'latitude': ('station', station_lats), 'longitude': ('station', station_lons)}
)
ds.to_netcdf(f'/gpfs/scratch1/shared/benitoli/RPs/fidelity_POT_notdetr_1981_2016_smalldiff/fidelity_{station}.nc')

