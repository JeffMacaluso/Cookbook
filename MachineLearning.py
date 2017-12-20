import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
