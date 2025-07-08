#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --ntasks=32
#SBATCH --partition=rome
#SBATCH --time=10:00:00

current_dir=$(pwd)

#---------------------------------------------------------------------------------------------------------------------------------------------------------
## INPUTS
#---------------------------------------------------------------------------------------------------------------------------------------------------------

current_date=$1
ensemble_member=$2
root_dir=$3
results_dir=$4
dates=$5

first_date=$(echo "$dates" | cut -d'/' -f1)
echo "First date: $first_date"

model_run=${current_date}_${ensemble_member}
model_run_dir="$root_dir/$model_run"

log_file="preGTSM_logs.txt"

ensemble_member_str=$(printf "%d" "$ensemble_member")

timeout_duration_SEAS5download=172800  # 2 days in seconds
timeout_duration_gtsm=302400  # 3.5 days in seconds

#---------------------------------------------------------------------------------------------------------------------------------------------------------
## Download SEAS5 and run GTSM for a specific date and ensemble member
#---------------------------------------------------------------------------------------------------------------------------------------------------------

echo "------------------------------------------------------- Event: $current_date // Ensemble member: $ensemble_member_str -------------------------------------------------------"
echo "Downloading SEAS5 for ensemble member $ensemble_member_str on date: $current_date"

SEAS5_file1="${root_dir}/SEAS5_${first_date}_ens_0_25.nc"
SEAS5_file2="${root_dir}/SEAS5_${first_date}_ens_26_50.nc"

# Check if both files exist
#if [[ -f "$SEAS5_file1" && -f "$SEAS5_file2" ]]; then      # If we would use forecasts from 2017, they have 51 ensemble members. Then use this.
if [[ -f "$SEAS5_file1" ]]; then
    echo "Files $SEAS5_file1 and $SEAS5_file2 exist."

    echo "Processing SEAS5 for GTSM, for ensemble member $ensemble_member_str on date: $current_date"
    conda run -n UNSEEN python /gpfs/work2/0/einf2224/paper3/scripts/forecasts/p1b_SEAS5_process4GTSM.py $current_date $ensemble_member $root_dir $dates

    SEAS5_post_file="${root_dir}/${model_run}/SEAS5_post_${current_date}_ens_${ensemble_member}*.nc"

    if ls $SEAS5_post_file 1> /dev/null 2>&1; then
        echo "File $SEAS5_post_file exists."
        
        echo "Preprocessing GTSM for ensemble member $ensemble_member_str on date: $current_date"
        conda run -n UNSEEN python p2_preprocess_GTSM.py $current_date $ensemble_member $root_dir
        
        echo "Running GTSM for ensemble member $ensemble_member_str on date: $current_date"
        # Run GTSM
        singularityFolder=/projects/0/einf2224/dflowfm_2022.04_fix20221108/delft3dfm_2022.04
        mdufile=gtsm_fine.mdu        
        #nPart=32 
        cd "${root_dir}/${model_run}/" || exit
        
        if timeout $timeout_duration_gtsm srun $singularityFolder/execute_singularity_snellius.sh -p 1 run_dflowfm.sh $mdufile; then        
            echo "GTSM run finished!"
            cd "$current_dir" || exit
            conda run -n UNSEEN python p2b_postprocess_GTSM.py $current_date $ensemble_member $root_dir $results_dir &&
            
            first_date_year=$(date -d "$current_date +1 month" +%Y-%m-%d)
            results_EU_file="${results_dir}/Europe/gtsm_EU_${first_date_year}_${ensemble_member}.nc"
            results_UK_file="${results_dir}/UK/gtsm_UK_${first_date_year}_${ensemble_member}.nc"
            #results_file="${results_dir}/gtsm_${first_date_year}_${ensemble_member}.nc"
            if [ -e "$results_EU_file" ] && [ -e "$results_UK_file" ]; then
            #if [ -e "$results_file" ]; then
                echo "Postprocessing finished!"
                echo "FIRST DATE YEAR: ${first_date_year}"
                echo "Files $results_EU_file and $results_UK_file exist."
                #echo "File $results_file exists."
                if [ -e "$model_run_dir" ]; then
                    echo "Deleting directory $model_run_dir."
                    rm -rf "$model_run_dir"
                else
                    echo "Directory $model_run_dir does not exist."
                fi
            else
                echo "Postprocessing did not finish successfully!"
                echo "File $results_EU_file and/or $results_UK_file does not exist."
            fi

        else
            echo "GTSM for $model_run did NOT finish successfully"
            echo "${model_run} FAILED!" >> GTSM_logs.txt
        fi  

    else
        echo "File $SEAS5_post_file does not exist!"
        echo "$current_date $ensemble_member_str SEAS5 postprocessed file does not exist" >> $log_file
    fi
else
    echo "File $SEAS5_file1 does not exist!" # "File $SEAS5_file1 and/or $SEAS5_file2 do not exist!"
    echo "$first_date $ensemble_member_str SEAS5 file does not exist" >> $log_file
fi
#else
#    echo "Download of SEAS5 for ensemble member $ensemble_member_str on date: $current_date failed!"
#    echo "$current_date $ensemble_member_str SEAS5 download failed" >> $log_file
#fi

echo "----------------------------------------------------------------------------------------------------------------------"
