import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

print("OS: ", sys.platform)
print("Python: ", sys.version)
print("NumPy: ", np.__version__)
print("Pandas: ", pd.__version__)

# Formatting for seaborn plots
sns.set_context("notebook", font_scale=1.1)
sns.set_style("ticks")

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

%matplotlib inline


### Missing Values
import missingno as msno  # Visualizes missing values
msno.matrix(df)
msno.heatmap(df)  # Co-occurrence of missing values

### Quick EDA report on dataframe
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
profile.get_rejected_variables(threshold=0.9)  # Rejected variables w/ high correlation
profile.to_file(outputfile="/tmp/myoutputfile.html")  # Saving report as a file
