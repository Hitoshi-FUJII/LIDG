# LIDG
Linearly independent descriptor generation (LIDG) program package for sparse and interpretable modeling

Note:
"LIDG" method is similar in name to "Ridge" regression, sure independent screening (SIS), and independent component analysis (ICA), but these are completely different methods.

## Latest version
Version 0.4 (2024 March 15)

Version 0.3 (2022 Jan 12)

Version 0.2 (2020 Nov 10)

Version 0.1 (2020 July 1)

Version 0.0 (2019 June 13)

## Changes
* version 0.3
    - Bug fix (sorted order of eq2 values in exhaustive search).
    - Add reading optin ("index_col" in .read() method).
    - Remove normalize optin in Elastic net calculation.  

* version 0.3
    - Display digits control
    - CSV file reading (added "sep" option in .read() method). 
    - Bug fix (outlier detection program in projection.py).
    
* version 0.2
    - Added a genetic algorithm code for tough model selections.
    
* version 0.1
    - Speedup and memory saving about Q2 (LOOCV) calclulation.
    - Added an exhaustive search method for model selection.

* version 0.0
    - Symmetrization of descriptors
    - Detection and removal of near multicollinearlities 
    - Linearly independentization of descriptor (design) matrix
    - Descriptor generation by direct product 
    - Descriptor generation by basic operations

## Required Packages
- python >= 3.7 (for ordered dictionary)
- matplotlib
- numpy
- pandas
- scipy (only for p-value calculation of t-test)
- scikt-learn (only for ElasticNet calculation)
- seaborn (only for display the results of model selections)

## Manual
see LIDG_manual.pdf

## License
LIDG is distributed under the MIT License.
Copyright (c) 2019 LIDG Development Team
