#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --partition=staging
#SBATCH --time=2-00:00:00

# Define the start and end dates. Loops are YEARLY! SO WE DOWNLOAD IN GROUPS OF MONTHS. ALL DECEMBERS, ALL JUNES
start_date=$1 #"2019-12-01"
end_date=$2 #"2020-01-01"
ensemble_member1=1  #0
ensemble_member2=2  #14

# Define the root directory
root_dir="/scratch-shared/benitoli"
echo "Root directory is set to: $root_dir"
results_dir="/gpfs/work2/0/einf2224/paper3/model_runs/gtsm_output"


# Define the job script template
job_template="job_template.sh"

# Initialize the counter
event_nr=0
current_date="$start_date"

# Log file
log_file="downloadSEAS5_logs.txt"

# Max duration to download data
timeout_duration_SEAS5download=172800  # 2 days in seconds

# Make a string with dates to download
start_timestamp=$(date -d "$start_date" +%s) # Convert start_date and end_date to timestamps 
end_timestamp=$(date -d "$end_date" +%s)
dates=""

# Loop to increment the date by 6 months
while [ "$(date -d "$current_date" +%s)" -lt "$end_timestamp" ]; do
  if [ -z "$dates" ]; then
    dates="$current_date"
  else
    dates="$dates/$current_date"
  fi
  current_date=$(date -d "$current_date + 12 months" +"%Y-%m-%d") # Increment the current_date by 12 months
done


SEAS5_file="${root_dir}/SEAS5_${start_date}_ens_0_25.nc"

if [[ -e "$SEAS5_file" ]]; then
    # Do something if the file exists
    echo "File $SEAS5_file exists. Proceeding with existing file."
else
    echo "File $SEAS5_file does not exist. Proceeding to download"
    if timeout $timeout_duration_SEAS5download conda run -n UNSEEN python /gpfs/work2/0/einf2224/paper3/scripts/forecasts/p1_ECMWF_web_api.py $dates $root_dir; then
        echo "Download of SEAS5 for ensemble member on date: $current_date was successful!"
    else
        echo "Download of SEAS5 for ensemble member on date: $current_date failed!"
        echo "$current_date SEAS5 download failed" >> $log_file
    fi
fi

# Start the loop only if the SEAS5 file exists
if [[ -e "$SEAS5_file" ]]; then
    echo "Starting loop as SEAS5 file $SEAS5_file is now available."
    current_date_job=$start_date

    echo "Dates: $dates"
    echo "Dates{}: ${dates}"


    # Loop over the dates
    while [[ "$current_date_job" < "$end_date" ]]; do
        echo "-------------------------------------- Current date: $current_date_job ------------------------------------------"
        #if timeout $timeout_duration_SEAS5download conda run -n UNSEEN python /gpfs/work2/0/einf2224/paper3/scripts/forecasts/p1_ECMWF_web_api.py $current_date $root_dir; then
        
        for ensemble_member in $(seq $ensemble_member1 $ensemble_member2); do
            ensemble_member_str=$(printf "%d" "$ensemble_member")
    
            event_nr=$((event_nr + 1))
    
            # Generate a unique job script for this iteration
            job_script="job_${current_date_job}_${ensemble_member_str}.sh"
            cp $job_template $job_script
    
            # Replace placeholders in the job script
            sed -i "s/\$1/$current_date_job/" $job_script
            sed -i "s/\$2/$ensemble_member/" $job_script
            sed -i 's#$3#'"'"$root_dir"'"'#g' $job_script
            sed -i 's#$4#'"'"$results_dir"'"'#g' $job_script
            sed -i 's#$5#'"'"$dates"'"'#g' $job_script
            
            cat $job_script
            
            # Submit the job script and capture the job ID
            job_id=$(sbatch $job_script | awk '{print $4}')
            echo "JOB ID: $job_id"
    
            while true; do
                # Get the job status
                job_status=$(squeue -j "$job_id" -h -o "%T")
                echo "JOB STATUS: $job_status"
                if [[ "$job_status" == "RUNNING" ]]; then
                    echo "Job is running. Deleting job script..."
                    rm "$job_script"  # Delete the job script once it starts running
                    break
                elif [[ "$job_status" == "PENDING" ]]; then
                    echo "Job is pending. Waiting for job to start..."
                elif [[ -z "$job_status" ]]; then
                    # Job finished or failed before it started running
                    echo "Job status is empty. The job has completed or failed before starting."
                    break
                fi
                sleep 30  # Check every 30 seconds
            done
    
        done
    
        # Move to the next date
        current_date_job=$(date -d "$current_date_job + 12 month" +%Y-%m-%d)
    done
else
    echo "SEAS5 file $SEAS5_file not available, skipping the loop."
fi    

# Print the total number of loops
echo "Total events processed: $event_nr"

