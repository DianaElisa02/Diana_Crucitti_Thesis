# Wealth Tax Simulation

This repository contains the full codebase for simulations and analysis conducted as part of the thesis:

Author: Diana Elisa Crucitti  
Institution: Utrecht University
Year: 2025

# File Overview

| File | Description |
|------|-------------|
| `dta_handling.py` | Utilities for reading `.dta` (Stata) files or also CVS files|
| `constants.py` | Central parameters and variables naming |
| `preprocessing.py` | Individual split, another function is included but not utilized in this simulation |
| `wealth_tax.py` | Core logic for computing wealth tax liability |
| `eff_typology.py` | Typology analysis: income/wealth classification |
| `reporting` | Summary statistics and descriptive output |
| `New_Simulation.py` | Main simulation script (with current cap policy) |
| `FlatTaxConterfactual.py` | Counterfactual scenario using a flat wealth tax rate |
| `README.md` | this file |

Further informations on the functions can be found in each file as well as the required packages

# Sensitivity Anlyses
At the top of the `New_Simulation.py` file, these type of code lines are found "USE_INDIVIDUAL = False". You have to selct TRUE to simulate the counterfactual scenarios.
