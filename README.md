# Python Event Studies

Python package for conducting event studies over the CRSP database.
Enable easy studies with standards methodologies but also implements the one in [the paper ]().

The package do not contains the data, you need to download the data from CRSP.

To install the package use the following command:

```bash
pip install py_event_studies
```

In order to use the package, here is how you can use it:

```python
import py_event_studies as pes

# Load the data (supports csv or parquet files)
# This step will take a little bit of time as it will not only load the data but also preprocess it by pivoting the table in order to be more efficient afterwards.
# It will save a cache file so if you reload the same path it will use the cache. If you changed the data pass the argument no_cache=True
pes.load_data('path/to/your/data.csv')

#If you want to use the Fama-French factors (optional, this step is however very fast as the data is not preprocessed)
pes.load_ff_factors('path/to/your/fama_french_factors.csv')

date = '20120816'

# Get the valid permnos at the date, not needed if you already have a list of permnos
valid_permnos = pes.get_valid_permno_at_date(date)

# Compute the event study for a portfolios
results = pes.compute(date, valid_permnos[np.array([1,10,50,23,35, 102, 55, 66, 548,1002])])

# Display the results statistics for standard tests, also available: cs_test_stats (cross sectionnal), bmp_test_stats (Boehmer, Musumeci and Poulsen (1991)), kp_test_stats (Kolari & Pynnönen (2010))
display(results.std_test_stats)

# In order to plot the prediction made by one of the model for a given cluster size (specify one even if it's a model that do not use one as here)
results.plot(5, 'FF5')

# Summary methods will print results of all tests for all models and all cluster sizes
results.summary()

# Finally you can save the results to an excel file with all results and statistics in different sheets to export it for further analysis
results.to_excel('path_to_save_results.xlsx')
```

## License

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
