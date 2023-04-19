import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

#################################################################################################################
### Plotting cluster averages and standard deviations around the average for two variables
# Assumes df has one row for each cluster and columns for the values and standard deviations

# Getting the colors for the clusters
cluster_color_map = pd.DataFrame({'cluster': df['cluster'].unique(),
                                  'color': sns.color_palette('Set1', df['cluster'].nunique())})

# Merging the data frames to have one with the x/y averages, standard deviations, and colors
data_to_plot = df.merge(cluster_color_map).merge(df_std)

# Plotting the average values
plt.figure(figsize=(10, 10))
plt.scatter(x=data_to_plot['x'], y=data_to_plot['y'], 
            color=data_to_plot['color'],
            s=data_to_plot['num_observations'])

# Plotting the standard deviation around each point
# Doing this in a loop to specify the color and size of the lines
for line in data_to_plot.iterrows():
    plt.errorbar(x=line[1]['x'],
                 y=line[1]['y'],
                 xerr=line[1]['x_std'],
                 yerr=line[1]['y_std'],
                 color=line[1]['color'],
                 elinewidth=line[1]['num_observations'] / 100,  # Scaling the lines to be smaller
                 alpha=0.25)

plt.title('Cluster Averages for X and Y')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

#################################################################################################################
#### Misc one liners

plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')  # Puts legend in the top right outside of the graph

# Forcing a discrete color palette for hue
# Docs: https://seaborn.pydata.org/tutorial/color_palettes.html
# Examples: https://chrisalbon.com/python/data_visualization/seaborn_color_palettes/
sns.scatterplot(x='x', y='y', hue='z', data=df, palette=sns.color_palette(palette='husl', n_colors=n))


# Formats y axis as a percentage
import matplotlib.ticker as mtick
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=0))
