import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import gc
from scipy.stats import kurtosis, skew, ttest_1samp, linregress

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Permutation test
def permutation_test(group_A, group_B, func, n_permutations=10000, seed=None):
    """
    Perform a permutation test to compare the means or standard deviations of two groups.

    Parameters:
    - group_A: array-like, first group of data
    - group_B: array-like, second group of data
    - parameter: str, 'mean' or 'std' to specify the parameter to test
    - n_permutations: int, number of permutations to perform
    - seed: int or None, random seed for reproducibility

    Returns:
    - observed_diff: float, observed difference in the specified parameter
    - p_value: float, permutation test p-value
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Check if dimensions match
    if group_A.shape[1] != group_B.shape[1]:
        raise ValueError("group_A and group_B must have the same number of columns")
    
    num_columns = group_A.shape[1]
    
    print('group A shape:', group_A.shape)
    print('group B shape:', group_B.shape)
    
    
    # Calculate the function for the observed diff. Without permutations
    if func == kurtosis:
        observed_diff = np.apply_along_axis(lambda x: func(x, fisher=False), 0, group_A) - np.apply_along_axis(lambda x: func(x, fisher=False), 0, group_B)
    else:
        observed_diff = np.apply_along_axis(func, 0, group_A) - np.apply_along_axis(func, 0, group_B)
    
    count=0   
    num_columns = group_A.shape[1]
    # Initialize array to store permuted differences
    permuted_diffs = np.zeros((n_permutations, num_columns))
    
    for i in range(n_permutations):
        print('Permutation:', count)
        sampled_indices = np.random.choice(group_B.shape[0], group_A.shape[0], replace=False)
        sampled_group_B = group_B[sampled_indices, :]
        print('samples_group_B shape:', sampled_group_B.shape)
        count += 1
    
        # Combine data from both groups
        combined_data = np.concatenate([group_A, sampled_group_B], axis=0)
        print('combined_data shape:', combined_data.shape)
        
        #for i in range(n_permutations):
        np.random.shuffle(combined_data)
        perm_group_A = combined_data[:len(group_A), :]
        perm_group_B = combined_data[len(group_A):, :]

        if func == kurtosis:
            permuted_diffs[i, :] = np.apply_along_axis(lambda x: func(x, fisher=False), 0, perm_group_A) - np.apply_along_axis(lambda x: func(x, fisher=False), 0, perm_group_B)
        else:
            permuted_diffs[i, :] = np.apply_along_axis(func, 0, perm_group_A) - np.apply_along_axis(func, 0, perm_group_B)
    
        print('permuted_diffs:', permuted_diffs)
        
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_diffs) >= np.abs(observed_diff), axis=0)
    print('permuted_diffs:', permuted_diffs)
    print('permuted_diffs shape:', permuted_diffs.shape)
    print('observed_diff:', observed_diff)
    print('p_value:', p_value)
    print('p_value shape:', p_value.shape)    
    
    return observed_diff, p_value
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Import ERA5 and SEAS5 and preprocess
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Open SEAS5 and ERA5 datasets
model_folder = sys.argv[5] #'Global_hindcasts'
directory_SEAS5=f'../../model_runs/SEAS5_statistics/independence_tests/{model_folder}/'
directory_ERA5=f'../../model_runs/SEAS5_statistics/independence_tests/Global_forecasts/'

combined_dataset=xr.open_dataset(directory_SEAS5 + 'SEAS5_merged.nc')
obs_combined_ds =  xr.open_dataset(directory_ERA5 + 'ERA5.nc').astype('float32')

# Sort datasets on time
obs_sorted_ds = obs_combined_ds.sortby('time')
ensemble_sorted_ds = combined_dataset.sortby('time')

# Slice datasets
if model_folder == 'Global_forecasts':
    start_date = '2017-01-01'
    end_date = '2023-12-31'
elif model_folder == 'Global_hindcasts':
    start_date = '1981-01-01'
    end_date = '2016-12-31'

obs_sliced_ds = obs_sorted_ds.sel(time=slice(start_date, end_date)) #.sel(time=slice('2017-01-01', None)) / .sel(time=slice('1994-01-01', None)) / .sel(time=slice('1979-01-01', None))
ensemble_sorted_ds_sliced = ensemble_sorted_ds.sel(time=slice(start_date, end_date))

# Clip ERA5 and SEAS5 grid to certain region, if wanted
lat_min = float(sys.argv[1])
lat_max = float(sys.argv[2])
lon_min = float(sys.argv[3])
lon_max = float(sys.argv[4])

if lat_min is not None and lat_max is not None and lon_min is not None and lon_max is not None:
    obs_ds = obs_sliced_ds.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
    ensemble_ds = ensemble_sorted_ds_sliced.sel(latitude=slice(lat_max, lat_min), longitude=slice(lon_min, lon_max))
else:
    obs_ds=obs_sliced_ds
    ensemble_ds=ensemble_sorted_ds_sliced

print('obs_ds:', obs_ds)
# Stack datasets to have gridcells instead of latitude and longitude
obs_stacked=obs_ds.stack(grid_cell=('latitude', 'longitude'))
ensemble_stacked=ensemble_ds.stack(grid_cell=('latitude', 'longitude'))


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Run functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

seasons = {
    'DJF': ('12', '01', '02'),
    'MAM': ('03', '04', '05'),
    'JJA': ('06', '07', '08'),
    'SON': ('09', '10', '11')
}

#for season_name, season_months in seasons.items():
for season_name, season_months in list(seasons.items())[3:4]:
    print(f"Processing season: {season_name}")
    
    # Filter data by season
    obs_season = obs_stacked.sel(time=obs_stacked['time.month'].isin([int(month) for month in season_months]))
    ensemble_season = ensemble_stacked.sel(time=ensemble_stacked['time.month'].isin([int(month) for month in season_months]))
        
    # Convert to numpy arrays and reshape ensemble 
    obs=obs_season['msl'].values
    ensemble=ensemble_season['msl'].values
    
    reshaped_ensemble = ensemble.reshape(-1, ensemble.shape[-1])
  
    
    # Run permutation tests for the season
    obs_diff_mean, p_value_mean = permutation_test(obs, reshaped_ensemble, func=np.mean, n_permutations=10000, seed=42)
    obs_diff_std, p_value_std = permutation_test(obs, reshaped_ensemble, func=np.std, n_permutations=100000, seed=42)

    
    # Save p-values as xarray Dataset
    mean_pvalue = p_value_mean.flatten()
    std_pvalue = p_value_std.flatten()
    mean_pvalue_reshaped = mean_pvalue.reshape(len(ensemble_ds['latitude']), len(ensemble_ds['longitude']))
    std_pvalue_reshaped = std_pvalue.reshape(len(ensemble_ds['latitude']), len(ensemble_ds['longitude']))


    result_pvalue = xr.Dataset({
        'mean_pvalue': (('latitude', 'longitude'), mean_pvalue_reshaped),
        'std_pvalue': (('latitude', 'longitude'), std_pvalue_reshaped),
    }, coords={
        'latitude': ensemble_ds['latitude'],
        'longitude': ensemble_ds['longitude']
    })
    
    print(f'result_pvalue for {season_name}:', result_pvalue)

    output_directory_pvalue = f'../../model_runs/SEAS5_statistics/independence_tests/{model_folder}/fidelity/pvalues_era5_long_seasons/'
    output_file_pvalue = os.path.join(output_directory_pvalue, f'SEAS5_fidelity_pvalue_{season_name}_lat_{lat_min}_{lat_max}_lon_{lon_min}_{lon_max}_v4.nc')
    encoding_pvalue = {var: {'dtype': 'float32'} for var in result_pvalue.data_vars}
    result_pvalue.to_netcdf(output_file_pvalue, encoding=encoding_pvalue)
    
    print(f"Saved p-values for season {season_name} to {output_file_pvalue}")
    