import xarray as xr
import numpy as np
from scipy.stats import spearmanr
import xskillscore as xs

# Calculation of Spearman correlation for each lead time, making timeseries for each ensemble member as a concatenation of all months

#---------------------------------------------------------------------------------------------------------------------------------------------
## Functions
#---------------------------------------------------------------------------------------------------------------------------------------------

def independence_test(ensemble, detrend=False, var_name="msl", ens_name="number", ld_name='leadtime'):
    lat_lon_shape = (ensemble.dims['latitude'], ensemble.dims['longitude'])
    n_grid_cells = np.prod(lat_lon_shape)

    n_ensembles=len(ensemble[ens_name])
    print('ensemble:', ensemble)
    flattened_data = ensemble[var_name].stack(grid_cell=('latitude', 'longitude'))
    print('flattened_data:', flattened_data)

    median_correlations = {}
    lower_correlations = {}
    upper_correlations = {}
    std_correlations = {}

    print('LEAD TIMES:', ensemble[ld_name])
    #for ld in ensemble[ld_name]:
    for ld in ensemble[ld_name]:
        print('LEAD TIME:', ld.values.item())
        data_arrays = []
        for mbr1 in ensemble[ens_name]:
            print('member 1:', mbr1.values.item())
            for mbr2 in ensemble[ens_name]:
                print('member 2:', mbr2.values.item())
                
                if mbr2 > mbr1:
                    predictant = flattened_data.sel(number=mbr1, leadtime=ld)
                    predictor = flattened_data.sel(number=mbr2, leadtime=ld)     

                    if detrend:
                        predictant_monthly_avg = predictant.groupby('time.month').mean(dim='time')
                        predictant_deseason = predictant - predictant_monthly_avg.sel(month=predictant['time.month'])                        
                        
                        predictor_monthly_avg = predictor.groupby('time.month').mean(dim='time')
                        predictor_deseason = predictor - predictor_monthly_avg.sel(month=predictor['time.month'])

                        correlation_coefficient = xs.spearman_r(predictant_deseason, predictor_deseason, dim='time')
                         
                    else:
                        correlation_coefficient = xs.spearman_r(predictant, predictor, dim='time')

                    if 'number' in correlation_coefficient.coords:
                        correlation_coefficient = correlation_coefficient.drop('number')
                    if 'leadtime' in correlation_coefficient.coords:
                        correlation_coefficient = correlation_coefficient.drop('leadtime')
                    
                    print('correlation_coefficient:', correlation_coefficient.isel(grid_cell=0))    
                    data_arrays.append(correlation_coefficient)
                              
        if data_arrays:
            print('data_arrays for lead time {}: {}'.format(ld.values.item(), data_arrays))
            concatenated_corr = xr.concat(data_arrays, dim='iteration')
            print('concatenated_corr for lead time {}: {}'.format(ld.values.item(), concatenated_corr))
            
            median_corr = concatenated_corr.median(dim='iteration')
            lower_quartile_corr = concatenated_corr.quantile(0.25, dim='iteration').drop_vars('quantile')
            upper_quartile_corr = concatenated_corr.quantile(0.75, dim='iteration').drop_vars('quantile')
            print('median_corr:', median_corr)
            
            median_correlations[ld.values.item()] = median_corr
            lower_correlations[ld.values.item()] = lower_quartile_corr
            upper_correlations[ld.values.item()] = upper_quartile_corr


    # Convert the dictionaries of median and std correlations into DataArrays
    median_corr_da = xr.Dataset(median_correlations).to_array(dim='leadtime')
    lower_corr_da = xr.Dataset(lower_correlations).to_array(dim='leadtime')
    upper_corr_da = xr.Dataset(upper_correlations).to_array(dim='leadtime')
    
    corr_ds = xr.Dataset({
        'median': median_corr_da,
        'lower_quart': lower_corr_da,
        'upper_quart': upper_corr_da
    })

    
    return corr_ds

#---------------------------------------------------------------------------------------------------------------------------------------------
## Input
#---------------------------------------------------------------------------------------------------------------------------------------------

model_folder = 'Global_hindcasts' #'Global_forecasts'
output_directory=f'../../model_runs/SEAS5_statistics/independence_tests/{model_folder}/'

combined_dataset=xr.open_dataset(output_directory + 'SEAS5_merged.nc')
print('combined_dataset:', combined_dataset)
sorted_combined_dataset = combined_dataset.sortby('time')
print('sorted_combined_dataset:', sorted_combined_dataset)

#---------------------------------------------------------------------------------------------------------------------------------------------
## Runs
#---------------------------------------------------------------------------------------------------------------------------------------------
# Calculate correlations
corr=independence_test(sorted_combined_dataset, detrend=True)
print('corr:', corr)

#---------------------------------------------------------------------------------------------------------------------------------------------
## Output
#---------------------------------------------------------------------------------------------------------------------------------------------
corr_unstacked = corr.unstack()    # To convert from gricell to latitude and longitude
print('corr_unstacked:', corr_unstacked)
corr_unstacked.to_netcdf(output_directory + 'SEAS5_independencetest_deseason_v4.nc')

