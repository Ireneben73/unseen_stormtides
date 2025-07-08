# unseen_stormtides
This repository contains the code for the reproduction of the paper:

Benito, I., Eilander, D., Kelder, T., Ward, P.J.,  Aerts, J.C.J.H., and Muis,S., 2024. Pooling Seasonal Forecast Ensembles to Estimate Storm Tide Return Periods in Extra-Tropical Regions.

In this study we use pool ensemble members of SEAS5 and use them as meteorological forcing todynamically simulate water levels using the Global Tide and Surge Model (GTSM). This allows us to create 525 synthetic years of storm tides that are subsequently used to estimate storm tide risk for Europe. 

The code consists of the following files and directories:
* **UNSEEN:** folder containing the scrips that do the independence and bias tests of SEAS5
   * **p2_UNSEEN_Merge_datasets.py:** script to merge the files from ERA5 and SEAS5 downloaded from CDS
   * **p3_UNSEEN_Independence_test.py:** script that performs the independence tests for SEAS5
   * **p5_UNSEEN_Fidelity_permutation_bootstrapping.py:** script that does the permutation test for all the year
   * **p5_UNSEEN_Fidelity_significance.py:** script that analyses the significance of the permutation test for all the year
   * **p5_UNSEEN_Fidelity_permutation_seasons.py:** script that does the permutation test for each season
   * **p5_UNSEEN_Fidelity_significance_seasons.py:** script that analyses the significance of the permutation test for each season
 
* **gtsm:** folder containing the scrips to run GTSM
   * **job_master_script_v2.sh:** bash script to modify the bash script job_template.sh to runs multiple GTSM workflows in parallel
   * **job_template:** bash script template that runs the GTSM workflow for a certain ensemble member and start date
   * **p2_preprocess_GTSM.py:** script to prepare the model files of GTSM forced with SEAS5
   * **p2_preprocess_GTSM_ERA5.py:** script to prepare the model files of GTSM forced with ERA5
   * **p2b_postprocess_GTSM:** script to postprocess the model files output of the GTSM forced with SEAS5
   * **p2b_postprocess_GTSM_ERA5:** script to postprocess the model files output of the GTSM forced with ERA5
