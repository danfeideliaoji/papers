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
  "ctree":              '#cccccc',
  "ctree + mask":       '#cccccc',
  "rpart":              '#377eb8',
  "rpart + mask":       '#377eb8',
  "Gaussian":           '#cc6633',
  "Gaussian + mask":    '#cc6633',
  "oor":                'r',
  "oor + mask":         'r',
  "mean":               '#ffccff',
  "mean + mask":        '#ffccff',
  "MIA":                '#00ff33',
}

FORESTS = ['DECISION TREE', 'RANDOM FOREST', 'XGBOOST', 'SVM', 'KNN']

# To obtain the measure of the variance below, run in R:
# > results <- make_data3bis(dim=9, size=200000)
# > var(results$y)

y_variances = {
    'mcar': 33,#10,
    'mnar': 33,
    'pred': 35,#10,
    'linearlinear': 25.4,
    'linearnonlinear': 1710,
    'nonlinearnonlinear': 1082,
}

# for name in ('mcar', 'mnar', 'pred', 'linearlinear', 'linearnonlinear',
#              'nonlinearnonlinear'):

# master_data is a dictionary containing all of the data we are going to plot
master_data = dict()

for name in ('mcar', 'mnar', 'pred'):
    data = pd.read_csv(f'results/scores_{name}.csv', header=0,
                    names=['index', 'score', 'method', 'forest'])

    # Knowing the variance of y, we can extract the R2
    data['R2'] = 1 - data['score'] / y_variances[name]
    # The fold number is encoded at the end of the name of the index
    data['fold'] = data['index'].str.extract('(\d+)$').astype(int)

    master_data[name] = data


height_ratios = [master_data['mcar'].query('forest == @forest')['method'].nunique()
                    for forest in FORESTS]

height = 10.5
width = 10.5

fig, axes = plt.subplots(5, 3, figsize=(width, height),
                        gridspec_kw=dict(height_ratios=height_ratios))

# values for minimum and maximum values for the x axis in all corresponding boxplots
xlim_min, xlim_max = 0.2, 1.0

missing_mechanisms = ['mcar', 'mnar', 'pred']

for col in range(len(missing_mechanisms)):
    for row in range(len(FORESTS)):
        ax = axes[row][col]
        forest = FORESTS[row]

        data_ = master_data[missing_mechanisms[col]]

        this_data = data_.query('forest == @forest')
        order = [k for k in color_mapping.keys()
                    if k in this_data['method'].unique()]
        # set x="rel_R2" to plot boxplots using values relative to the mean of R2 values
        # set x="R2" to plot using the same axis
        g = sns.boxplot(x="R2", y="method",
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
        ax.set_xlim(xlim_min, xlim_max)

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
plt.savefig(f'figures/boxplot_grid.pdf')
