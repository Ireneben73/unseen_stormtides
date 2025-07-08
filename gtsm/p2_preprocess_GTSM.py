#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:25:23 2023
@author: benitoli
"""

import sys
import numpy as np
import os
import shutil
import fnmatch

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
sys.path.append('src')
import templates
from distutils.dir_util import copy_tree
import glob 
import pandas as pd
import shutil

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Input settings from the bash script
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

date=sys.argv[1]
ens_member=sys.argv[2]

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Directories
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

root_dir=sys.argv[3]
templatedir=f"/gpfs/work2/0/einf2224/paper3/scripts/gtsm/gtsm_template/model_input_template"
modelfilesdir=f"/gpfs/work2/0/einf2224/paper3/scripts/gtsm/gtsm_template/model_files_common"
rundir=f"{root_dir}/{date}_{ens_member}" # directory where each gtsm model will be ran

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Copy GTSM files to each model folder
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# copy template files to rundir
print("copying ",templatedir," to ",rundir)
copy_tree(templatedir,rundir)
    
# copy static model files to rundir
print("copying ",modelfilesdir," to ",rundir)
copy_tree(modelfilesdir, rundir)


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Modify template GTSM files according to Case Study and Model Configuration
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Modify model config file
#---------------------------
tstart = datetime.strptime(date, "%Y-%m-%d")
tstart_str = tstart.strftime("%Y%m%d")

# Find the hours from which the output of his should be saved. We are throwing the first month for spinup 
if tstart.month == 12:
    next_month = 1
    next_year = tstart.year + 1
else:
    next_month = tstart.month + 1
    next_year = tstart.year

first_day_this_month = tstart
first_day_next_month = datetime(next_year, next_month, 1)
his_start = int((first_day_next_month - first_day_this_month).total_seconds())
print('num_seconds_start:', his_start)

# Calculate the tstop as the 7 months after the start_date
last_day_7th_month = (tstart + relativedelta(months=7)).replace(day=1)
end_date = last_day_7th_month - timedelta(seconds=3600)
time_difference = end_date - tstart
his_end = int(time_difference.total_seconds()) # In seconds for the his file
run_end = int(his_end / 3600) # In hours for tstop
print('num_seconds:', his_end)
print('num_hours:', run_end)

obsfile="obs_points_europe_100km_UK_2km.xyn" #"selected_output_new_unique_noreg.xyn"


keywords_MDU={'REFDATE':tstart_str,'TSTART':str(0),'TSTOP':str(run_end), 'OBSFILE':obsfile , 'HISINT':f'3600 {his_start} {his_end}', 'MAPINT':str(0)} 
templates.replace_all(os.path.join(rundir,"gtsm_fine.mdu.template"), os.path.join(rundir,"gtsm_fine.mdu"),keywords_MDU,'%')

# Replace all partitioned files so that we do not need to partition again
mdu_template_files=glob.glob(os.path.join(rundir,"gtsm_fine_00*.mdu.template"))
for mdu_template_file in mdu_template_files:
    # Construct the output file name with the .mdu extension
    mdu_file = os.path.splitext(mdu_template_file)[0] #+ ".mdu"
    # Replace the keywords in the input file and write to the output file
    templates.replace_all(os.path.join(rundir,mdu_template_file), os.path.join(rundir,mdu_file), keywords_MDU, '%')

## Modify external forcings file
#--------------------------------
# define settings based on meteorological forcing
keywords_EXT={'METEOFILE_SEAS5_WX':f'SEAS5_post_{date}_ens_{ens_member}_u10.nc', 'METEOFILE_SEAS5_WY':f'SEAS5_post_{date}_ens_{ens_member}_v10.nc', 'METEOFILE_SEAS5_P':f'SEAS5_post_{date}_ens_{ens_member}_msl.nc'}
templates.replace_all(os.path.join(rundir,"gtsm_fine.ext.template"),os.path.join(rundir,"gtsm_fine.ext"),keywords_EXT,'%') 
