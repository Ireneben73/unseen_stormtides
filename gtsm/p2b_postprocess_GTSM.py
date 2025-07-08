#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:25:23 2023
@author: benitoli
"""

import os
from datetime import datetime
import xarray as xr
import numpy as np
import sys
import glob

date=sys.argv[1]
ens_member=sys.argv[2]
root_dir=sys.argv[3]
SEAS_dir = f'{root_dir}/{date}_{ens_member}'
results_dir=sys.argv[4]


# Open the dataset and remove the first timestep
gtsm_netfile = os.path.join(SEAS_dir, 'output', 'gtsm_fine_0000_his.nc')
original_dataset = xr.open_dataset(gtsm_netfile).isel(time=slice(1, None))
dataset = original_dataset[['waterlevel', 'bedlevel']]

# Get first date year
first_date_year = str(dataset.time.min().dt.strftime('%Y-%m-%d').item())

# Drop stations that fall dry
min_waterlevel = dataset.waterlevel.min(dim='time')
max_waterlevel = dataset.waterlevel.max(dim='time')
condition = (min_waterlevel == dataset.bedlevel) | (max_waterlevel == dataset.bedlevel)
dataset_filtered = dataset.where(~condition, drop=True)

def attrib(dataset, frequency):
    dataset.attrs={'title': f'{frequency} timeseries of total water levels forced with SEAS5. Forecast initializd: {date}. Ensemble member: {ens_member}', 
              'summary': 'This dataset has been produced with the Global Tide and Surge Model (GTSM) version 4.1. GTSM was forced with wind a pressure fields from the seasonal forecast from ECMWF SEAS5', 
              'date_created': str(datetime.utcnow()) + ' UTC', 
              'contact': 'i.benito.lazaro@vu.nl',
              'institution': 'IVM - VU Amsterdam', #include partners later on., 
              'source': 'GTSMv4.1 forced with SEAS5',
              'keywords': 'extratropical cyclone; SEAS5; water level; storm surge; tides; global tide and surge model;', 
              'geospatial_lat_min': dataset.station_y_coordinate.min().round(3).astype(str).item(), 
              'geospatial_lat_max': dataset.station_y_coordinate.max().round(3).astype(str).item(), 
              'geospatial_lon_min': dataset.station_x_coordinate.min().round(3).astype(str).item(), 
              'geospatial_lon_max': dataset.station_x_coordinate.max().round(3).astype(str).item(), 
              'geospatial_lat_units': 'degrees_north',
              'geospatial_lat_resolution': 'point',
              'geospatial_lon_units': 'degrees_east', 
              'geospatial_lon_resolution': 'point',
              'geospatial_vertical_min': dataset['waterlevel'].min().round(3).astype(str).item(), 
              'geospatial_vertical_max': dataset['waterlevel'].max().round(3).astype(str).item(),
              'geospatial_vertical_units': 'm', 
              'geospatial_vertical_positive': 'up',
              'time_coverage_start': str(dataset.time.min().dt.strftime('%Y-%m-%d %H:%M:%S').item()), 
              'time_coverage_end': str(dataset.time.max().dt.strftime('%Y-%m-%d %H:%M:%S').item())}
    return dataset

def process_dataset(dataset, frequency, outfile):
    """Process the dataset by cleaning and adding attributes, then save to a NetCDF file."""
    # Drop unnecessary variables and dimensions
    #dataset = dataset.drop_vars(['station_name', 'station_geom', 'station_geom_node_count', 'timestep', 'wgs84'])
    #dataset = dataset.drop_dims(['station_geom_nNodes'])

    # Set attributes for key variables
    dataset.waterlevel.attrs = {
        'long_name': 'sea_surface_height_above_mean_sea_level',
        'units': 'm',
        'short_name': 'waterlevel',
        'description': 'Total water level resulting from the combination of barotropic tides and surges and mean sea-level'
    }
    dataset.station_y_coordinate.attrs = {
        'units': 'degrees_north',
        'short_name': 'latitude',
        'long_name': 'latitude'
    }
    dataset.station_x_coordinate.attrs = {
        'units': 'degrees_east',
        'short_name': 'longitude',
        'long_name': 'longitude'
    }
    dataset.time.attrs = {
        'axis': 'T',
        'long_name': 'time',
        'short_name': 'time'
    }
    dataset['waterlevel'] = dataset.waterlevel.round(3)

    dataset = dataset.assign_coords({'stations': dataset.stations})

    # Update vertical min/max in attributes
    dataset.attrs['geospatial_vertical_min'] = dataset.waterlevel.min().round(3).astype(str).item()
    dataset.attrs['geospatial_vertical_max'] = dataset.waterlevel.max().round(3).astype(str).item()

    # Add attributes
    dataset = attrib(dataset, frequency)

    # Set encoding for saving
    encoding = {
        'stations': {'dtype': 'uint16', 'complevel': 3, 'zlib': True},
        'station_y_coordinate': {'dtype': 'int32', 'scale_factor': 0.001, '_FillValue': -999, 'complevel': 3, 'zlib': True},
        'station_x_coordinate': {'dtype': 'int32', 'scale_factor': 0.001, '_FillValue': -999, 'complevel': 3, 'zlib': True},
        'waterlevel': {'dtype': 'int16', 'scale_factor': 0.001, '_FillValue': -999, 'complevel': 3, 'zlib': True}
    }

    # Save to NetCDF
    dataset[['waterlevel']].to_netcdf(outfile, encoding=encoding)



# Save dataset for Europe with daily maxima data
dataset_europe_dailymax = dataset_filtered.resample(time='1D').max()
europe_dir = os.path.join(results_dir, 'Europe')
os.makedirs(europe_dir, exist_ok=True)
outfile_europe=os.path.join(europe_dir, f'gtsm_EU_{first_date_year}_{ens_member}.nc')
process_dataset(dataset_europe_dailymax, 'Daily maxima', outfile_europe)

# Save dataset for UK with hourly data
min_x = -11
max_x = 2
min_y = 49.8
max_y = 59.8

dataset_uk = dataset_filtered.where(
    (dataset_filtered.station_x_coordinate >= min_x) &
    (dataset_filtered.station_x_coordinate <= max_x) &
    (dataset_filtered.station_y_coordinate >= min_y) &
    (dataset_filtered.station_y_coordinate <= max_y),
    drop=True
)
uk_dir = os.path.join(results_dir, 'UK')
os.makedirs(uk_dir, exist_ok=True)
outfile_uk=os.path.join(uk_dir, f'gtsm_UK_{first_date_year}_{ens_member}.nc')
process_dataset(dataset_uk, '1-hourly', outfile_uk)
