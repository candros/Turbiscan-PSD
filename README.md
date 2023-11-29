# Turbiscan-PSD
Source code for the OSC PSD model described by Andros et al (in press)

The folder "raw turbiscan data" contains the comma-delimited data files obtained from the Turbiscan Tower instrument for 29 different soils. These are the same data files used the train the "short-term experiment" model, as described by Andros et al (in press).

The folder "turbiscan model" contains pickled versions of the final 10-minute model (separate models to predict silt, sand and clay) as described by Andros et al (in press).

The folder "utils" contains a python script "osc_model.py" with the helper functions needed to run and /or the OSC PSD model.

The notebook "OSC PSD Notebook" contains the code necessary to either train a new OSC PSD model or to predict the PSD of the samples run on the Turbiscan, as descirbed by Andros et al (in press).

The excel spreadsheet "training_pipette_psd.xlsx" contains all the pipette data used to train the model described by Andros et al (in press). 
