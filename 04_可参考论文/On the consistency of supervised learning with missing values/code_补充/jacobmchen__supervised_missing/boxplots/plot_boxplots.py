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
for name in ('mcar', 'mnar', 'pred'):
  data = pd.read_csv(f'results/scores_{name}.csv', header=0,
                  names=['index', 'score', 'method', 'forest'])

  # Knowing the variance of y, we can extract the R2
  data['R2'] = 1 - data['score'] / y_variances[name]
  # The fold number is encoded at the end of the name of the index
  data['fold'] = data['index'].str.extract('(\d+)$').astype(int)


  for drop_mask in (False, True):
    if drop_mask:
        data_ = data[~data['method'].str.contains("mask")]
    else:
        data_ = data.copy()

    # this line of code finds the mean of each of the R2 and makes that the "center"
    # line to compare other values
    data_['rel_R2'] = data_.groupby(['fold', 'forest'])['R2'].apply(
        lambda df: df - df.mean())

    height_ratios = [data_.query('forest == @forest')['method'].nunique()
                    for forest in FORESTS]

    #from matplotlib import colors
    #color_mapping = {k: colors.to_rgb(v) for k, v in color_mapping.items()}

    height = 6.3 if drop_mask else 10.5
    width = 4.6 if drop_mask else 5.6

    fig, axes = plt.subplots(5, 1, figsize=(width, height),
                            gridspec_kw=dict(height_ratios=height_ratios))

    # values for minimum and maximum values for the x axis in all corresponding boxplots
    xlim_min, xlim_max = 0.2, 1.0

    for forest, ax in zip(FORESTS, axes):
        print('ax', ax)
        this_data = data_.query('forest == @forest')
        print('this_data\n', this_data)
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
        if name == 'mcar':
            # ax.set_xlim(-.095, .2)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'mnar':
            # ax.set_xlim(-.33, .17)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'pred':
            # ax.set_xlim(-.09, .2)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'linearlinear':
            # ax.set_xlim(-.24, .18)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'linearnonlinear':
            # ax.set_xlim(-.2, .2)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'nonlinearnonlinear':
            # ax.set_xlim(-50, 55)
            ax.set_xlim(xlim_min, xlim_max)
        sns.despine(bottom=True, left=False)
        for i in range(len(order)):
            if i % 2:
                ax.axhspan(i - .5, i + .5, color='.9', zorder=-2)

    plt.tight_layout(pad=.01, h_pad=2)

    # We need to do the ticks in a second pass, as they get modified by
    # the tight_layout
    for forest, ax in zip(FORESTS, axes):
        if name == 'mcar':
            # ax.set_xlim(-.15, .2)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'mnar':
            # ax.set_xlim(-.33, .17)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'pred':
            # ax.set_xlim(-.09, .2)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'linearlinear':
            # ax.set_xlim(-.07, .07)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'linearnonlinear':
            # ax.set_xlim(-.09, .09)
            ax.set_xlim(xlim_min, xlim_max)
        elif name == 'nonlinearnonlinear':
            # ax.set_xlim(-.32, .32)
            ax.set_xlim(xlim_min, xlim_max)

        """
        the following code makes the x-axis labels have positive or negative values
        depending on whether the values are less than or greater than the mean of the
        R2 values in the dataframe
        """
        # this_data = data.query('forest == @forest')
        # ticks = ax.get_xticks()
        # ticklabels = list()
        # for t in ticks:
        #     if t < 0:
        #         ticklabels.append(format_float(t))
        #     elif t > 0:
        #         ticklabels.append('+' + format_float(t))
        #     else:
        #         ticklabels.append('$%s$' % format_float(
        #                           this_data['R2'].mean()))
        # ax.set_xticklabels(ticklabels)
        # end of above inclusion

    plt.tight_layout(pad=.01, h_pad=2)
    mask_str = '_no_mask' if drop_mask else ''
    plt.savefig(f'figures/boxplot_{name}{mask_str}.pdf')
