# Turbiscan-PSD
Source code for the OSC PSD model described by Andros et al (2024)

The notebook "OSC PSD Notebook" contains the code necessary to either train a new OSC PSD model or to predict the PSD of the samples run on the Turbiscan, as descirbed by Andros et al (2024).

the folder "run files" should be used as a repository for turbiscan files that the user wishes to examine using the OSC PSD model. The OSC PSD notebook will look in this directory for input files if training is set to false.

The folder "raw turbiscan data" contains the comma-delimited data files obtained from the Turbiscan Tower instrument for 29 different soils. These are the same data files used the train the "short-term experiment" model, as described by Andros et al (2024). These are included in case the user wishes to re-train the model or rerun the training routine.

The folder "turbiscan model" contains pickled versions of the final 10-minute model (separate models to predict silt, sand and clay) as described by Andros et al (2024).

The folder "utils" contains a python script "osc_model.py" with the helper functions needed to run and /or the OSC PSD model.

The excel spreadsheet "training_pipette_psd.xlsx" contains all the pipette data used to train the model described by Andros et al (2024). 

The file "environment.yml" contains the libraries and dependencies at the time of writing the code.
