import xarray as xr
import numpy as np
import glob
import sys

meteo=sys.argv[1]

datasets = []

# Loop through the files and read each one
for file in glob.glob('/gpfs/scratch1/shared/benitoli/RPs/fidelity_POT_notdetr_1981_2016_smalldiff/fidelity_*.nc'):
    # Open the dataset
    ds = xr.open_dataset(file)
    print('STATUS:', ds.status.values)
    # Add a station coordinate to each dataset (use filename or index to differentiate stations)
    station_id = file.split('_')[-1].split('.')[0]  # Use station ID from filename, e.g., "3267"
    ds = ds.assign_coords(station=("station", [station_id]))  # Add station ID as a coordinate
    
    # Append the dataset to the list
    datasets.append(ds)

# Concatenate all datasets along the 'station' dimension
rps = xr.concat(datasets, dim='station')

# Print the result to check
print(rps)
print('status:', np.unique(rps.status.values))


#print(rps.status.dtypes)
encoding = {
    'status': {
        'dtype': 'S1',
    }
}
rps.to_netcdf(f'/gpfs/work2/0/einf2224/paper3/model_runs/RPs/fidelity_POT_notdetr_1981_2016_smalldiff.nc', encoding=encoding)
print('status2:', np.unique(rps.status.values))

fidelity = xr.open_dataset(f'/gpfs/work2/0/einf2224/paper3/model_runs/RPs/fidelity_POT_notdetr_1981_2016_smalldiff.nc')



