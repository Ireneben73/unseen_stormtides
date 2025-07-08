import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, genextreme, genpareto
import scipy.stats as stats
import sys

#--------------------------------------------------------------------------------------------------------------------
## 0. Functions
#--------------------------------------------------------------------------------------------------------------------

#def pot(data, return_periods, seas5, percentile=99.5, n_bootstrap=1000, ci=95):
def pot(data, return_periods, percentile=99.5, n_bootstrap=1000, ci=95):
    """
    Compute return levels using the Peaks Over Threshold (POT) method with a Generalized Pareto Distribution (GPD)
    and estimate confidence intervals using bootstrapping.

    Parameters:
    - data (xarray.DataArray or numpy array): The time series data.
    - return_periods (numpy array): Array of return periods for which return levels will be calculated.
    - seas5 (bool): Whether to process data as an ensemble with multiple members.
    - percentile (float, optional): The percentile threshold for exceedance (default: 99.5).

    Returns:
    - return_levels_gdp (numpy array): Estimated return levels.
    """
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
    
    data = data.compute()
    threshold = np.percentile(data.values, percentile)
    combined_waterlevels = []
    
    if meteo == 'SEAS5':
    #if seas5:
        for ensemble in data.ensemble_member.values:
            data_ens = data.sel(ensemble_member=ensemble)
            exceedances = data_ens.where(data_ens > threshold, drop=True)
            exceedance_times = exceedances.time.values
            exceedance_values = exceedances.values
            _, declustered_waterlevels = decluster_exceedances(exceedance_times, exceedance_values)
            combined_waterlevels.extend(declustered_waterlevels)
    else:
        exceedances = data.where(data > threshold, drop=True)
        exceedance_times = exceedances.time.values
        exceedance_values = exceedances.values
        _, declustered_waterlevels = decluster_exceedances(exceedance_times, exceedance_values)
        combined_waterlevels.extend(declustered_waterlevels)
    
    combined_waterlevels = np.array(combined_waterlevels)
    sorted_extremes = np.sort(combined_waterlevels)[::-1]
    ranks = np.arange(1, len(sorted_extremes) + 1)
    return_periods_empirical = (len(sorted_extremes) + 1) / ranks
    
    lambda_factor = 365  # Number of days per year
    time_len = len(data.time)
    
    if meteo == 'SEAS5':
    #if seas5:
        ensemble_member_len = len(data.ensemble_member)
        years = (time_len * ensemble_member_len) / lambda_factor  # Approximate total years in dataset
        print('years:', years)
    else:
        years = (time_len) / lambda_factor  # Approximate total years in dataset
        print('years:', years)
        
    lambda_exceedance = len(combined_waterlevels) / years 
    params = stats.genpareto.fit(combined_waterlevels - threshold, floc=0)   
    probabilities_lambda = 1 - (1 / (lambda_exceedance * return_periods))
    return_levels_gdp = threshold + stats.genpareto.ppf(probabilities_lambda, *params)
    
    shape, loc, scale = params
    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        resampled_data = np.random.choice(combined_waterlevels, size=len(combined_waterlevels), replace=True)
        params_boot = stats.genpareto.fit(resampled_data - threshold, floc=0)
        bootstrap_estimates.append(threshold + stats.genpareto.ppf(probabilities_lambda, *params_boot))
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    lower_bound = np.percentile(bootstrap_estimates, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(bootstrap_estimates, 100 - (100 - ci) / 2, axis=0)
    
    return shape, loc, scale, lambda_exceedance, threshold, return_levels_gdp, lower_bound, upper_bound 


#--------------------------------------------------------------------------------------------------------------------
## 1. Loop per station
#--------------------------------------------------------------------------------------------------------------------

idx=int(sys.argv[1])
print('idx:', idx)
meteo=sys.argv[2]
print('meteo:', meteo)
    
#--------------------------------------------------------------------------------------------------------------------
## 2. Open input data (ERA5 or SEAS5)
#--------------------------------------------------------------------------------------------------------------------

if meteo == 'SEAS5':
    ensemble_members = range(0, 15)  # adjust as needed for your ensemble range
    gtsm_ens_append = []
    
    # Loop through ensemble members
    for ensemble_member in ensemble_members:
        gtsm_seas5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/Europe/gtsm_EU_*_{ensemble_member}.nc')
        
        # Add an attribute for the ensemble member
        gtsm_ens = gtsm_seas5.assign_coords(ensemble_member=ensemble_member)
        gtsm_ens_append.append(gtsm_ens)
    
    # Concatenate along the new ensemble dimension
    gtsm_ens_concat = xr.concat(gtsm_ens_append, dim="ensemble_member")
    
    gtsm_station_wl = gtsm_ens_concat["waterlevel"].isel(stations=idx)
    print('gtsm_station_wl:', gtsm_station_wl)

elif meteo == 'ERA5':
    gtsm_era5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/ERA5/Europe/gtsm_EU_*.nc')
    gtsm_station_wl = gtsm_era5["waterlevel"].isel(stations=idx).compute()

else:
    print('Meteo does not exist')
#--------------------------------------------------------------------------------------------------------------------
## 3. POT Fit 
#--------------------------------------------------------------------------------------------------------------------

return_periods = np.array([10, 40, 100, 500, 1000])
shape, loc, scale, lambda_exceedance, threshold, return_levels_gdp, lower_bound, upper_bound  = pot(gtsm_station_wl, return_periods)
#shape, loc, scale, lambda_exceedance, threshold, return_levels_gdp, lower_bound, upper_bound  = pot(gtsm_station_wl, return_periods, seas5=True)


#--------------------------------------------------------------------------------------------------------------------
## 3. Prepare data
#--------------------------------------------------------------------------------------------------------------------

station_name_list = [gtsm_station_wl.station_name.item().decode('utf-8').strip()]  # Wrap in list
station_x_coord_list = [gtsm_station_wl.station_x_coordinate.item()]  # Wrap X coordinate in list
station_y_coord_list = [gtsm_station_wl.station_y_coordinate.item()]  # Wrap Y coordinate in list

# Create an xarray.Dataset with the correct dimensions and coordinates
gdp = xr.Dataset(
    {
        "return_level": (["return_period"], return_levels_gdp),
        "lower_bound": (["return_period"], lower_bound),
        "upper_bound": (["return_period"], upper_bound),
        "gdp_params": (["param_type"], [loc, scale, shape, lambda_exceedance, threshold]),
    },
    coords={
        "return_period": return_periods,
        "param_type": ["loc", "scale", "shape", "lambda_exceedance", "threshold"],
        "stations": (["stations"], np.array([idx], dtype=np.int16)),  # Single station index
        "station_name": (["stations"], station_name_list),  # Station name as a list
        "station_x_coordinate": (["stations"], station_x_coord_list),  # X coordinate as list
        "station_y_coordinate": (["stations"], station_y_coord_list),  # Y coordinate as list
    },
)

print('gdp:', gdp)
gdp.to_netcdf(f'/gpfs/scratch1/shared/benitoli/RPs/stations_{meteo}/{meteo}_GDP_RPs_{idx}.nc')