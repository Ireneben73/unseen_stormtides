import xarray as xr
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, genextreme, genpareto
import scipy.stats as stats
from pyextremes import get_extremes, plot_mean_residual_life, plot_parameter_stability
import random


factor = 10

#--------------------------------------------------------------------------------------------------------------------
## Functions
#--------------------------------------------------------------------------------------------------------------------
    
def pot(data, return_periods, seas5, percentile=99.5, n_bootstrap=1000, ci=95):
    """
    Compute return levels using the Peaks Over Threshold (POT) method with a Generalized Pareto Distribution (GPD)
    and estimate confidence intervals using bootstrapping.

    Parameters:
    - data (xarray.DataArray or numpy array): The time series data.
    - return_periods (numpy array): Array of return periods for which return levels will be calculated.
    - seas5 (bool): Whether to process data as an ensemble with multiple members.
    - percentile (float, optional): The percentile threshold for exceedance (default: 99.5).
    - n_bootstrap (int, optional): Number of bootstrap resamples (default: 1000).
    - ci (float, optional): Confidence interval percentage (default: 95).

    Returns:
    - sorted_extremes (numpy array): Sorted declustered exceedances.
    - return_periods_empirical (numpy array): Empirical return periods.
    - return_levels_gdp (numpy array): Estimated return levels.
    - return_levels_ci (tuple of numpy arrays): Lower and upper confidence intervals for return levels.
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
    
    if seas5:
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
    '''
    combined_waterlevels = np.array(combined_waterlevels)
    sorted_extremes = np.sort(combined_waterlevels)[::-1]
    ranks = np.arange(1, len(sorted_extremes) + 1)
    return_periods_empirical = (len(sorted_extremes) + 1) / ranks
    '''
    lambda_factor = 365  # Number of days per year
    time_len = len(data.time)
    if seas5:
        ensemble_member_len = len(data.ensemble_member)
        years = (time_len * ensemble_member_len) / lambda_factor  # Approximate total years in dataset
        print('years:', years)
    else:
        years = (time_len) / lambda_factor  # Approximate total years in dataset
        print('years:', years)
        
    def get_return_periods(x, extremes_rate, a=0.0):
        b = 1.0 - 2.0*a
        ranks = (len(x) + 1) - stats.rankdata(x, method="average")
        freq = ((ranks - a) / (len(x) + b)) * extremes_rate
        rps = 1 / freq
        return ranks, rps
        
    lambda_exceedance = len(combined_waterlevels) / years 
    combined_waterlevels = np.array(combined_waterlevels)
    sorted_extremes = np.sort(combined_waterlevels)[::-1]
    ranks, return_periods_empirical = get_return_periods(sorted_extremes.tolist(), extremes_rate=lambda_exceedance)
    
    #combined_waterlevels = np.array(combined_waterlevels)
    #sorted_extremes = np.sort(combined_waterlevels)[::-1]
    #ranks = np.arange(1, len(sorted_extremes) + 1)
    #return_periods_empirical = (len(sorted_extremes) + 1) / ranks
    
    params = stats.genpareto.fit(combined_waterlevels - threshold, floc=0)   
    probabilities_lambda = 1 - (1 / (lambda_exceedance * return_periods))
    return_levels_gdp = threshold + stats.genpareto.ppf(probabilities_lambda, *params)

    
    bootstrap_estimates = []
    for _ in range(n_bootstrap):
        resampled_data = np.random.choice(combined_waterlevels, size=len(combined_waterlevels), replace=True)
        params_boot = stats.genpareto.fit(resampled_data - threshold, floc=0)
        bootstrap_estimates.append(threshold + stats.genpareto.ppf(probabilities_lambda, *params_boot))
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    lower_bound = np.percentile(bootstrap_estimates, (100 - ci) / 2, axis=0)
    upper_bound = np.percentile(bootstrap_estimates, 100 - (100 - ci) / 2, axis=0)
    
    return sorted_extremes, return_periods_empirical, return_levels_gdp, lower_bound, upper_bound


#--------------------------------------------------------------------------------------------------------------------
## 1. Concatenate ensembles for SEAS5
#--------------------------------------------------------------------------------------------------------------------

ensemble_members = range(0, 15)  # adjust as needed for your ensemble range
annual_max_datasets = []

# Loop through ensemble members
for ensemble_member in ensemble_members:
    gtsm_seas5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/Europe/gtsm_EU_*_{ensemble_member}.nc')
    
    # Calculate annual maxima
    gtsm_annualmax = gtsm_seas5#.resample(time="3D").max()
    
    # Add an attribute for the ensemble member
    gtsm_annualmax = gtsm_annualmax.assign_coords(ensemble_member=ensemble_member)
    annual_max_datasets.append(gtsm_annualmax)

# Concatenate along the new ensemble dimension
gtsm_ensemble_annualmax = xr.concat(annual_max_datasets, dim="ensemble_member")

#--------------------------------------------------------------------------------------------------------------------
## 2. Open ERA5
#--------------------------------------------------------------------------------------------------------------------

gtsm_era5 = xr.open_mfdataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output/ERA5/Europe/gtsm_EU_*.nc')

#--------------------------------------------------------------------------------------------------------------------
## 3. Extract stations
#--------------------------------------------------------------------------------------------------------------------

cities_df = pd.read_csv('cities_v2.csv', sep=';')
cities_df = cities_df.sort_values(by='City', ascending=True)
print('cities_df:', cities_df)

#--------------------------------------------------------------------------------------------------------------------
## 4. Loop per station
#--------------------------------------------------------------------------------------------------------------------

# BEST
stations_dict={
    #'38': 'hoekvanholland-hoe-nld-cmems', # Try hoekvanholland-hoe-nld-cmems
    '1468': 'bangor-ban-gbr-cmems',
    #'1559': 'howth-how-irl-cmems', # Try howth-how-irl-cmems
    '43': 'bake_c_scharhorn-3905-deu-wsv', # Try bake_c_scharhorn-3905-deu-wsv
    '577': 'bournemouth-bou-gbr-cmems', # Try swanage_pier-swp-gbr-cco
    '1175': 'liverpool-liv-gbr-cmems',
    '354': 'drogden-dro-dnk-cmems', # Try skanor-ska-swe-cmems
    '65': 'bergen-bgo-nor-nhs',
    #'377': 'kielholtenau-kie-deu-cmems',
    #'5': 'cascais-209a-prt-uhslc',
    #'503': 'brighton-btn-gbr-cco',
    #'8': 'leixoes-lei-prt-cmems',
    #'17': 'saint_jean_de_luz_socoa-95-fra-refmar',    # FAR
    '33': 'lehavre_60minute-leh-fra-cmems',
    #'3029': 'leith-lei-gbr-bodc', #'leith-lei-gbr-cmems',
    '3105': 'northshields-nor-gbr-cmems',
    '3358': 'herne_bay-hby-gbr-cco', #'sheerness-she-gbr-cmems', #'sheerness-she-gbr-bodc', 
    '20': 'iledaix_60minute-ile-fra-cmems', # Try iledaix_60minute-ile-fra-cmems
}

station_ids = cities_df['Station_id'].tolist()

# Set up the figure and subplots

rows, cols = 6, 3
fig, axs = plt.subplots(rows, cols, figsize=(12, 17)) 
axs = axs.flatten()

for ix, idx in list(enumerate(station_ids)):
    print('idx:', idx)

    #--------------------------------------------------------------------------------------------------------------------
    ## Empirical RPs & GDP RPs ERA5 & SEAS5
    #--------------------------------------------------------------------------------------------------------------------    
    station_seas5=gtsm_ensemble_annualmax["waterlevel"].isel(stations=idx)
    station_era5=gtsm_era5["waterlevel"].isel(stations=idx)
    
    return_periods_gev = np.linspace(5, 525, 1000)
    sorted_extremes_seas5, return_periods_empirical_seas5, return_levels_gdp_seas5, lower_bound_seas5, upper_bound_seas5 = pot(station_seas5, return_periods_gev, seas5=True)
    sorted_extremes_era5, return_periods_empirical_era5, return_levels_gdp_era5, lower_bound_era5, upper_bound_era5 = pot(station_era5, return_periods_gev, seas5=False)
        
    #--------------------------------------------------------------------------------------------------------------------
    ## Figure
    #--------------------------------------------------------------------------------------------------------------------

    axs[ix].scatter(return_periods_empirical_era5, sorted_extremes_era5, color='blue', s=10, zorder=99, label='Empirical fit ERA5')
    axs[ix].scatter(return_periods_empirical_seas5, sorted_extremes_seas5, color='firebrick', s=10, label='Empirical fit SEAS5')

    if str(idx) in stations_dict:
        gesla_station = stations_dict.get(str(idx))
        gesla_rps_path = f'/gpfs/work2/0/einf2224/paper3/scripts/validation/gesla_rps/{gesla_station}_RPs_newstations.csv'
        
        if os.path.exists(gesla_rps_path):
            gesla_rps = pd.read_csv(gesla_rps_path)    
            print('gesla_rps:', gesla_rps)
            axs[ix].scatter(gesla_rps['return_period'], gesla_rps['waterlevel'], color='black', s=10, zorder=100, label='Empirical fit GESLA')

    axs[ix].plot(return_periods_gev, return_levels_gdp_era5, label='POT-GPD fit ERA5', color='blue')  
    axs[ix].plot(return_periods_gev, return_levels_gdp_seas5, label='POT-GPD fit SEAS5', color='firebrick') 
      
    axs[ix].fill_between(
        return_periods_gev,
        lower_bound_era5,
        upper_bound_era5,
        color='blue',
        alpha=0.3,
        label='95% CI ERA5',
        edgecolor='none'
    )
    
   
    axs[ix].fill_between(
        return_periods_gev,
        lower_bound_seas5,
        upper_bound_seas5,
        color='firebrick',
        alpha=0.3,
        label='95% CI SEAS5',
        edgecolor='none'
    )

    #--------------------------------------------------------------------------------------------------------------------
    ## Continue with the figure
    #-------------------------------------------------------------------------------------------------------------------- 
    axs[ix].set_xscale('log')  # Use log scale for return periods
    axs[ix].set_xlim(left=1) 
    axs[ix].grid(False)
    
    city_name = cities_df[cities_df['Station_id'] == idx]['City'].values[0]
    axs[ix].set_title(city_name)

    if ix % cols == 0:  # First column
        axs[ix].set_ylabel('Storm tide (m)')
    if ix // cols == rows - 1:  # Last row
        axs[ix].set_xlabel('Return Period (years)')

    if ix == 0:  # Custom legend handling for the first subplot
        handles, labels = axs[ix].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)#, bbox_to_anchor=(0.5, -0.05))

plt.subplots_adjust(left=0.05, right=0.98, top=0.98, bottom=0.09)
fig.savefig('figures/Fig1_cities_POT_CI_gesla_v2.png', dpi=300)
    

