import os
import sys
import xarray as xr
import numpy as np
import pandas as pd
import gc
from scipy.stats import kurtosis, skew, ttest_1samp

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Functions
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# False discovery rate    
def benjamini_hochberg(pvalue_array, alpha=0.05):
    """
    Apply the Benjamini-Hochberg procedure to control the false discovery rate (FDR).

    Parameters:
    - pvalue_array: array-like, the array of p-values to be adjusted
    - alpha: float, significance level (default is 0.05)

    Returns:
    - significant_pvalues: array-like, boolean array indicating which p-values are significant
    """
    # Number of tests
    m = len(pvalue_array)
    
    # Sort p-values and keep track of the original indices
    sorted_indices = np.argsort(pvalue_array)
    sorted_pvalues = pvalue_array[sorted_indices]
    
    # Calculate the FDR threshold using the Benjamini-Hochberg procedure
    thresholds = np.array([(i / m) * alpha for i in range(1, m + 1)])
    
    # Determine which p-values are significant
    significant = sorted_pvalues <= thresholds
    
    # Map significant results back to the original shape
    significant_pvalues = np.zeros_like(pvalue_array, dtype=bool)
    significant_pvalues[sorted_indices] = significant
    
    return significant_pvalues
    

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Import p-values of fidelity
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
model_folder = sys.argv[1]

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
    p_values_ds =  xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/SEAS5_statistics/independence_tests/{model_folder}/fidelity/pvalues_era5_long_seasons/*{season_name}*_v4.nc')
    print('p_values_ds:', p_values_ds)
    
    p_values_df = p_values_ds.to_dataframe().reset_index()
    p_values_df['lat_lon'] = p_values_df['latitude'].astype(str) + '-' + p_values_df['longitude'].astype(str)
    p_values_df_cleaned = p_values_df.drop_duplicates(subset='lat_lon')
    p_values_df_cleaned = p_values_df_cleaned.drop(columns='lat_lon')
    p_values_df_cleaned = p_values_df_cleaned.set_index(['latitude', 'longitude']).to_xarray()
    
    print('p_values_df_cleaned:', p_values_df_cleaned)
    
    # Stack dataset to get gridcells
    p_values_stacked=p_values_df_cleaned.stack(grid_cell=('latitude', 'longitude'))
    p_values_mean = p_values_stacked.mean_pvalue.values
    p_values_std = p_values_stacked.std_pvalue.values
    
    print('p values max:', p_values_stacked.mean_pvalue.values.max())
    print('p values min:', p_values_stacked.mean_pvalue.values.min())
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Run functions to account for spatial correlations in the p-values
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    significant_mean = benjamini_hochberg(p_values_mean, alpha=0.1)        # TO DO: This should be 0.05x2 I think, for FDR
    significant_num_mean = significant_mean.astype(int)
    significant_std = benjamini_hochberg(p_values_std, alpha=0.1)
    significant_num_std = significant_std.astype(int)
    
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ## Save outputs as xarrayDataset and export to netcdf
    #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Save significance
    significant_num_mean = significant_num_mean.flatten()
    significant_num_std = significant_num_std.flatten()
    #significant_num_skew = significant_num_skew.flatten()
    #significant_num_kurt = significant_num_kurt.flatten()
    significant_num_mean_reshaped = significant_num_mean.reshape(len(p_values_df_cleaned['latitude']), len(p_values_df_cleaned['longitude']))
    significant_num_std_reshaped = significant_num_std.reshape(len(p_values_df_cleaned['latitude']), len(p_values_df_cleaned['longitude']))
    
    result_sign = xr.Dataset({
        'mean_sign': (('latitude', 'longitude'), significant_num_mean_reshaped),
        'std_sign': (('latitude', 'longitude'), significant_num_std_reshaped),
    }, coords={
        'latitude': p_values_df_cleaned['latitude'],
        'longitude': p_values_df_cleaned['longitude']
    })
    
    print('result_sign:', result_sign)
    
    output_directory_sign=f'../../model_runs/SEAS5_statistics/independence_tests/{model_folder}/fidelity/significance_era5_seasons/'
    output_file_sign = os.path.join(output_directory_sign, f'SEAS5_fidelity_sign_{season_name}_v4.nc')
    encoding_sign = {var: {'dtype': 'float32'} for var in result_sign.data_vars}
    result_sign.to_netcdf(output_file_sign, encoding=encoding_sign)
