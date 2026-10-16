import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def format_float(x):
    fmt = '%.2f'
    for i in range(-1, 1):
        if abs(x) > 10**i:
            fmt = '%%.%if' % (1 - i)
    for i in range(2, 4):
        if abs(x) > 10**i:
            x = np.round(x, i)
            fmt = '%i'
    return fmt % x


sns.set_style("whitegrid",
        {'axes.spines.left': False,
         'axes.spines.bottom': False,
         'axes.spines.right': False,
         'axes.spines.top': False,
         'text.color': 'k',
         'xtick.color': 'k',
         'ytick.color': 'k',
         })

plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['xtick.major.pad'] = .02

color_mapping = {
  "ctree(train)":              '#cccccc',
  "ctree + mask(train)":       '#cccccc',
  "rpart(train)":              '#377eb8',
  "rpart + mask(train)":       '#377eb8',
  "Gaussian(train)":           '#cc6633',
  "Gaussian + mask(train)":    '#cc6633',
  "oor(train)":                'r',
  "oor + mask(train)":         'r',
  "mean(train)":               '#ffccff',
  "mean + mask(train)":        '#ffccff',
  "MIA(train)":                '#00ff33',
  "ctree(test)":              '#cccccc',
  "ctree + mask(test)":       '#cccccc',
  "rpart(test)":              '#377eb8',
  "rpart + mask(test)":       '#377eb8',
  "Gaussian(test)":           '#cc6633',
  "Gaussian + mask(test)":    '#cc6633',
  "oor(estn)":                'r',
  "oor + mask(test)":         'r',
  "mean(test)":               '#ffccff',
  "mean + mask(test)":        '#ffccff',
  "MIA(test)":                '#00ff33',
}

FORESTS = ['DECISION TREE', 'RANDOM FOREST', 'XGBOOST', 'SVM', 'KNN']

# for name in ('mcar', 'mnar', 'pred', 'linearlinear', 'linearnonlinear',
#              'nonlinearnonlinear'):

# master_data is a dictionary containing all of the data we are going to plot
master_data = dict()

for name in ('mcar', 'mnar', 'pred'):
    data_train = pd.read_csv(f'results/train_times_{name}.csv', header=0,
                           names=['index', 'time', 'method', 'forest'])
    
    data_test = pd.read_csv(f'results/test_times_{name}.csv', header=0,
                           names=['index', 'time', 'method', 'forest'])
    
    for i in range(len(data_train)):
        data_train.at[i, 'method'] = data_train.at[i, 'method'] + '(train)'
        data_test.at[i, 'method'] = data_test.at[i, 'method'] + '(test)'

    data = pd.concat([data_train, data_test], ignore_index=True)

    master_data[name] = data


height_ratios = [master_data['mcar'].query('forest == @forest')['method'].nunique()
                    for forest in FORESTS]

height = 20.5
width = 10.5

"""
NOTE: to change plot between 1 column and 3 columns, there are three sections of the
code that you need to change that are labeled with #!
"""

#! if only plot the first column: change first two parameters to 5, 1
#! if plot all three columns: change first two parameters to 5, 3
fig, axes = plt.subplots(5, 1, figsize=(width, height),
                        gridspec_kw=dict(height_ratios=height_ratios))

# values for minimum and maximum values for the x axis in all corresponding boxplots
ensemble_xlim_min, ensemble_xlim_max = 0, 0.6
decision_tree_xlim_min, decision_tree_xlim_max = 0, 0.03
svm_xlim_min, svm_xlim_max = 0, 0.06
knn_xlim_min, knn_xlim_max = 0, 0.06

#! if only plot the first column: uncomment the first line
#! if plot all three columns: uncomment the second line
missing_mechanisms = ['mcar']
# missing_mechanisms = ['mcar', 'mnar', 'pred']

for col in range(len(missing_mechanisms)):
    for row in range(len(FORESTS)):
        #! if only plot the first column: uncomment the first line
        #! if plot all three columns: uncomment the second line
        ax = axes[row]
        # ax = axes[row][col]

        forest = FORESTS[row]

        data_ = master_data[missing_mechanisms[col]]

        this_data = data_.query('forest == @forest')
        order = [k for k in color_mapping.keys()
                    if k in this_data['method'].unique()]
        g = sns.boxplot(x="time", y="method",
                        data=this_data,
                        ax=ax, fliersize=0, palette=color_mapping,
                        order=order,
                        saturation=1,
                        boxprops=dict(ec='k'),
                        medianprops=dict(color='k'))
        ax.set_title(forest, pad=0.5, size=15, loc='right')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.axvline(0, color='.8', zorder=0, linewidth=3)
        # to do: reduce the amount of ticks in the decision tree plots
        if forest == 'RANDOM FOREST' or forest == 'XGBOOST':
            ax.set_xlim(ensemble_xlim_min, ensemble_xlim_max)
        elif forest == 'DECISION TREE':
            ax.set_xlim(decision_tree_xlim_min, decision_tree_xlim_max)

            # Specify the number of ticks you want on the x-axis
            num_ticks = 3  # Adjust this value based on your preference

            # Calculate tick positions
            tick_positions = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num_ticks)

            # Set the x-axis ticks
            ax.set_xticks(tick_positions)

            # manually set the first tick to be 0.0 for better readability
            current_labels = [label.get_text() for label in ax.get_xticklabels()]
            current_labels[0] = '0.0'

            ax.set_xticklabels(current_labels)
        elif forest == 'SVM':
            ax.set_xlim(svm_xlim_min, svm_xlim_max)
        elif forest == 'KNN':
            ax.set_xlim(knn_xlim_min, knn_xlim_max)

        if col > 0:
            ax.set_yticklabels([])

        sns.despine(bottom=True, left=False)
        for i in range(len(order)):
            if i % 2:
                ax.axhspan(i - .5, i + .5, color='.9', zorder=-2)

        plt.tight_layout(pad=.01, h_pad=2)

# We need to do the ticks in a second pass, as they get modified by
# the tight_layout
# for missing_mechanism, forest, ax in zip(missing_mechanisms, FORESTS, axes):
#     ax.set_xlim(xlim_min, xlim_max)

plt.tight_layout(pad=.01, h_pad=2)
plt.savefig(f'figures/train_test_times_boxplot_grid.pdf')
